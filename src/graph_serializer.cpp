#include <zerograd/graph_serializer.hpp>
#include <zerograd/graph_analyzer.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace zerograd
{
    std::string GraphSerializer::escape_json_string(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch(c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\t': out += "\\t"; break;
                default: out += c; break;
            }
        }
        return out;
    }

    bool GraphSerializer::has_cycle_dfs(
        const std::shared_ptr<Tensor>& node,
        std::unordered_set<std::shared_ptr<Tensor>>& gray,
        std::unordered_set<std::shared_ptr<Tensor>>& black
    )
    {
        if (black.count(node))
            return false;
        if (gray.count(node))
            return true;

        gray.insert(node);
        for (auto& child : node->_children) {
            if (has_cycle_dfs(child, gray, black))
                return true;
        }
        gray.erase(node);
        black.insert(node);

        return false;
    }

    bool GraphSerializer::validate_dag(const std::shared_ptr<Tensor>& root)
    {
        std::unordered_set<std::shared_ptr<Tensor>> gray;
        std::unordered_set<std::shared_ptr<Tensor>> black;
        return !has_cycle_dfs(root, gray, black);
    }

    void GraphSerializer::serialize(const std::shared_ptr<Tensor>& root, const std::string& output_path)
    {
        if (!validate_dag(root)) {
            throw std::runtime_error("GraphSerializer::serialize: graph contains a cycle, refusing to dump.");
        }
        
        std::vector<std::shared_ptr<Tensor>> topo = Tensor::get_topo_order(root);

        GraphAnalyzer analyzer;
        auto lifetimes = analyzer.dry_forward(root);

        std::unordered_map<std::shared_ptr<Tensor>, std::size_t> node_id;
        for (auto& t: topo) {
            node_id[t] = t->birth_step;
        }

        std::ostringstream out;
        out << "{\n  \"nodes\": [\n";

        for (std::size_t i = 0; i < topo.size(); ++i) {
            auto& t = topo[i];
            auto [birth, death, size_bytes] = lifetimes.at(t);

            out << "    {\n";
            out << "      \"id\": " << node_id[t] << ",\n";
            out << "      \"op\": \"" << escape_json_string(t->_op.empty() ? "leaf" : t->_op) << "\",\n";
            out << "      \"shape\": [";
            for (std::size_t j = 0; j < t->shape.size(); ++j) {
                out << t->shape[j] << (j + 1 < t->shape.size() ? ", " : "");
            }
            out << "],\n";
            out << "      \"birth_step\": " << birth << ",\n";
            out << "      \"death_step\": " << death << ",\n";
            out << "      \"size_bytes\": " << size_bytes << "\n";
            out << "    }" << (i + 1 < topo.size() ? "," : "") << "\n";
        }
        out << "  ],\n  \"edges\": [\n";

        std::vector<std::pair<std::size_t, std::size_t>> edges;
        for (auto& t : topo) {
            for (auto& child : t->_children) {
                edges.emplace_back(node_id[child], node_id[t]);
            }
        }

        for (std::size_t i = 0; i < edges.size(); ++i) {
            out << "    { \"from\": " << edges[i].first << ", \"to\": " << edges[i].second << " }"
                << (i + 1 < edges.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";

        std::ofstream file(output_path);
        if (!file) {
            throw std::runtime_error("GraphSerializer::serialize: could not open output file: " + output_path);
        }
        file << out.str();
    }
}