// SPDX-License-Identifier: Apache-2.0
#include "vae_encoder.h"
#include "filesystem_utils.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
namespace h3image
{
namespace
{
struct Node
{
    std::string type, name;
    std::vector<std::string> inputs, outputs;
    std::map<int, std::string> parameters;
    int get(std::string& error, int key, int fallback = 0) const
    {
        auto it = parameters.find(key);
        return it == parameters.end() ? fallback : parse_i32(error, it->second);
    }
};
std::vector<Node> read_graph(std::string& error, const std::string& path)
{
    std::ifstream file(path);
    if (!file)
    {
        error = "cannot open VAE graph: " + path;
        return {};
    }
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    std::vector<Node> nodes;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::istringstream row(line);
        Node node;
        int ni, no;
        row >> node.type >> node.name >> ni >> no;
        node.inputs.resize(ni);
        node.outputs.resize(no);
        for (auto& name : node.inputs)
            row >> name;
        for (auto& name : node.outputs)
            row >> name;
        std::string field;
        while (row >> field)
        {
            auto equal = field.find('=');
            node.parameters[parse_i32(error, field.substr(0, equal))] = field.substr(equal + 1);
            if (!error.empty())
                return {};
        }
        nodes.push_back(std::move(node));
    }
    return nodes;
}
ncnn::Mat crop(const ncnn::Mat& input, int y, int x, int height, int width)
{
    ncnn::Mat result(width, height, input.c);
    for (int c = 0; c < input.c; c++)
        for (int row = 0; row < height; row++)
            std::memcpy(result.channel(c).row(row), input.channel(c).row(y + row) + x, width * sizeof(float));
    return result;
}
ncnn::Mat extract_convolution(std::string& error, H3ModelSet& models, const Node& node, const ncnn::Mat& input)
{
    auto* net = models.prepare(error, Component::VAEEncoder);
    if (!net)
        return {};
    VulkanWorkspace workspace(*net);
    ncnn::Extractor extractor = net->create_extractor();
    workspace.bind(extractor);
    ncnn::Mat output;
    if (extractor.input(node.inputs[0].c_str(), input) || extractor.extract(node.outputs[0].c_str(), output))
    {
        error = "VAE convolution extraction failed: " + node.name;
        return {};
    }
    return unpack_float32(error, output);
}
ncnn::Mat convolution(std::string& error, H3ModelSet& models, const Node& node, const ncnn::Mat& input, int tile_size)
{
    const int kw = node.get(error, 1), kh = node.get(error, 11, kw), sw = node.get(error, 3, 1),
              sh = node.get(error, 13, sw), dw = node.get(error, 2, 1), dh = node.get(error, 12, dw);
    if (!error.empty())
        return {};

    if (node.get(error, 4) || node.get(error, 14) || node.get(error, 15) || node.get(error, 16))
    {
        error = "VAE tiled convolution expects explicit padding";
        return {};
    }
    const int ew = (kw - 1) * dw + 1, eh = (kh - 1) * dh + 1, ow = (input.w - ew) / sw + 1,
              oh = (input.h - eh) / sh + 1, tw = std::max(1, tile_size / sw), th = std::max(1, tile_size / sh);
    if (ow <= tw && oh <= th)
        return extract_convolution(error, models, node, input);
    ncnn::Mat output(ow, oh, node.get(error, 0));
    if (!error.empty())
        return {};

    for (int y = 0; y < oh; y += th)
        for (int x = 0; x < ow; x += tw)
        {
            const int h = std::min(th, oh - y), w = std::min(tw, ow - x);
            ncnn::Mat patch = crop(input, y * sh, x * sw, (h - 1) * sh + eh, (w - 1) * sw + ew);
            ncnn::Mat result = extract_convolution(error, models, node, patch);
            if (!error.empty())
                return {};

            if (result.dims != 3 || result.c != output.c || result.h != h || result.w != w)
            {
                error = "VAE convolution tile shape mismatch";
                return {};
            }
            for (int c = 0; c < output.c; c++)
                for (int j = 0; j < h; j++)
                    std::memcpy(output.channel(c).row(y + j) + x, result.channel(c).row(j), w * sizeof(float));
        }
    return output;
}
}
ncnn::Mat encode_vae_tiled(std::string& error, H3ModelSet& models, const ncnn::Mat& image, int tile_size)
{
    if (image.dims != 3 || image.c != 3 || image.h % 16 || image.w % 16 || tile_size < 16)
    {
        error = "invalid VAE encoder image/tile size";
        return {};
    }
    auto nodes = read_graph(error, models.model_prefix(Component::VAEEncoder) + ".ncnn.param");
    if (!error.empty())
        return {};

    std::unordered_map<std::string, int> uses;
    for (const Node& node : nodes)
        for (const auto& name : node.inputs)
            uses[name]++;
    std::unordered_map<std::string, ncnn::Mat> values;
    values["in0"] = image;
    for (const Node& node : nodes)
    {
        if (node.type == "Input")
            continue;
        if (node.type == "Split")
        {
            for (const auto& out : node.outputs)
            {
                values[out] = checked_at(error, values, node.inputs[0]);
                if (!error.empty())
                    return {};
            }
        }
        else
        {
            ncnn::Mat result;
            if (node.type == "Convolution")
            {
                result = convolution(error, models, node, checked_at(error, values, node.inputs[0]), tile_size);
                if (!error.empty())
                    return {};
            }
            else
            {
                auto* net = models.prepare(error, Component::VAEEncoder);
                if (!net)
                    return {};
                VulkanWorkspace workspace(*net);
                ncnn::Extractor extractor = net->create_extractor();
                workspace.bind(extractor);
                for (const auto& name : node.inputs)
                {
                    if (extractor.input(name.c_str(), checked_at(error, values, name)))
                    {
                        error = "invalid tiled VAE input: " + name;
                        return {};
                    }
                    if (!error.empty())
                        return {};
                }
                if (extractor.extract(node.outputs[0].c_str(), result))
                {
                    error = "tiled VAE extraction failed: " + node.name;
                    return {};
                }
                result = unpack_float32(error, result);
                if (!error.empty())
                    return {};
            }
            values[node.outputs[0]] = result;
        }
        for (const auto& name : node.inputs)
            if (--uses[name] == 0)
                values.erase(name);
    }
    return checked_at(error, values, "out0");
}
}
