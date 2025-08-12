#ifndef __TIMETREE_HPP__
#define __TIMETREE_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <cmath>
#include <cstdint>
#include <memory>


class TimeNode
{
public:
    int64_t timestamp;
    std::string arbitrary_node_info;
    std::shared_ptr<TimeNode> left;
    std::shared_ptr<TimeNode> right;
    int height;

    TimeNode(int64_t t, const std::string &info)
        : timestamp(t), arbitrary_node_info(info), left(nullptr), right(nullptr), height(1) {}
};

class TimeTree
{
public:
    std::shared_ptr<TimeNode> m_root = nullptr;

    TimeTree()
    {
        std::cout << "Initialized empty TimeTree." << std::endl;
    }

    TimeTree(const std::string &filename)
    {
        // Check if filename is a binary
        std::ifstream in(filename, std::ios::binary);
        if (in.is_open())
        {
            this->load(filename);
            return;
        }

        // Files are txt files, so let's build the tree from scratch
        m_root = buildAVLTree(filename);
    }

    std::shared_ptr<TimeNode> get(int64_t timestamp, int64_t threshold = 1000)
    {
        return findClosest(m_root, timestamp, threshold);
    }

    bool save(const std::string &binfile) const
    {
        std::ofstream out(binfile, std::ios::binary);
        if (!out.is_open())
            return false;
        save_node(out, m_root);
        return true;
    }

    static std::shared_ptr<TimeTree> load(const std::string &binfile)
    {
        std::ifstream in(binfile, std::ios::binary);
        if (!in.is_open())
            return nullptr;
    auto tree = std::make_shared<TimeTree>();
        tree->m_root = load_node(in);
        return tree;
    }

    int getHeight(const std::shared_ptr<TimeNode> &node)
    {
        return node ? node->height : 0;
    }

    int getBalanceFactor(const std::shared_ptr<TimeNode> &node)
    {
        return node ? getHeight(node->left) - getHeight(node->right) : 0;
    }

    int countLeafNodes(const std::shared_ptr<TimeNode> &node)
    {
        if (!node)
            return 0;
        if (!node->left && !node->right)
            return 1;
        return countLeafNodes(node->left) + countLeafNodes(node->right);
    }

    int getTreeDepth(const std::shared_ptr<TimeNode> &node)
    {
        if (!node)
            return 0;
        return 1 + std::max(getTreeDepth(node->left), getTreeDepth(node->right));
    }

    int getTotalNodes(const std::shared_ptr<TimeNode> &node)
    {
        if (!node)
            return 0;
        return 1 + getTotalNodes(node->left) + getTotalNodes(node->right);
    }

    std::shared_ptr<TimeNode> appendAVLTree(const std::string &timestamp_filepath)
    {
        std::cout << "Appending to AVLtree from " << timestamp_filepath << std::endl;
        m_root = buildAVLTree(timestamp_filepath, m_root);
    }

    std::shared_ptr<TimeNode> buildAVLTree(const std::string &timestamp_filepath, std::shared_ptr<TimeNode> root = nullptr)
    {
        std::cout << "Building AVLtree from " << timestamp_filepath << std::endl;
        std::ifstream file(timestamp_filepath);
        if (!file.is_open())
        {
            std::cout << "ERROR: Cannot open file: " << timestamp_filepath << std::endl;
            return nullptr;
        }

        std::string line;
        while (getline(file, line))
        {
            std::istringstream iss(line);
            std::string prefix, timestamp_str, frameidx;
            if (std::getline(iss, prefix, '_') &&
                std::getline(iss, timestamp_str, '_') &&
                std::getline(iss, frameidx))
            {
                int64_t timestamp = std::stoll(timestamp_str);
                root = insert(root, timestamp, frameidx);
            }
        }

        return root;
    }    

protected:
    std::shared_ptr<TimeNode> rotateRight(std::shared_ptr<TimeNode> y)
    {
        auto x = y->left;
        auto T2 = x->right;

        x->right = y;
        y->left = T2;

        y->height = std::max(getHeight(y->left), getHeight(y->right)) + 1;
        x->height = std::max(getHeight(x->left), getHeight(x->right)) + 1;

        return x;
    }

    std::shared_ptr<TimeNode> rotateLeft(std::shared_ptr<TimeNode> x)
    {
        auto y = x->right;
        auto T2 = y->left;

        y->left = x;
        x->right = T2;

        x->height = std::max(getHeight(x->left), getHeight(x->right)) + 1;
        y->height = std::max(getHeight(y->left), getHeight(y->right)) + 1;

        return y;
    }

    std::shared_ptr<TimeNode> insert(std::shared_ptr<TimeNode> root, int64_t timestamp, const std::string &frameidx)
    {
        if (!root)
            return std::make_shared<TimeNode>(timestamp, frameidx);

        if (timestamp < root->timestamp)
            root->left = insert(root->left, timestamp, frameidx);
        else if (timestamp > root->timestamp)
            root->right = insert(root->right, timestamp, frameidx);
        else
            return root;

        root->height = 1 + std::max(getHeight(root->left), getHeight(root->right));
        int balance = getBalanceFactor(root);

        if (balance > 1 && timestamp < root->left->timestamp)
            return rotateRight(root);
        if (balance < -1 && timestamp > root->right->timestamp)
            return rotateLeft(root);
        if (balance > 1 && timestamp > root->left->timestamp)
        {
            root->left = rotateLeft(root->left);
            return rotateRight(root);
        }
        if (balance < -1 && timestamp < root->right->timestamp)
        {
            root->right = rotateRight(root->right);
            return rotateLeft(root);
        }

        return root;
    }

    std::shared_ptr<TimeNode> findClosest(std::shared_ptr<TimeNode> root, int64_t target, int64_t threshold)
    {
        std::shared_ptr<TimeNode> closest = nullptr;
        int64_t minDiff = std::numeric_limits<int64_t>::max();

        while (root)
        {
            int64_t diff = std::abs(root->timestamp - target);
            if (diff < minDiff)
            {
                minDiff = diff;
                closest = root;
            }
            root = (target < root->timestamp) ? root->left : root->right;
        }

        return (closest && minDiff <= threshold) ? closest : nullptr;
    }

    static void save_node(std::ofstream &out, const std::shared_ptr<TimeNode> &node)
    {
        if (!node)
            return;

        out.write(reinterpret_cast<const char *>(&node->timestamp), sizeof(int64_t));
        size_t len = node->arbitrary_node_info.size();
        out.write(reinterpret_cast<const char *>(&len), sizeof(size_t));
        out.write(node->arbitrary_node_info.data(), len);

        bool has_left = node->left != nullptr;
        bool has_right = node->right != nullptr;
        out.write(reinterpret_cast<const char *>(&has_left), sizeof(bool));
        out.write(reinterpret_cast<const char *>(&has_right), sizeof(bool));

        if (has_left)
            save_node(out, node->left);
        if (has_right)
            save_node(out, node->right);
    }

    static std::shared_ptr<TimeNode> load_node(std::ifstream &in)
    {
        if (in.eof())
            return nullptr;

        int64_t timestamp;
        size_t len;
        std::string frameidx;
        bool has_left, has_right;

        in.read(reinterpret_cast<char *>(&timestamp), sizeof(int64_t));
        in.read(reinterpret_cast<char *>(&len), sizeof(size_t));
        frameidx.resize(len);
        in.read(&frameidx[0], len);
        in.read(reinterpret_cast<char *>(&has_left), sizeof(bool));
        in.read(reinterpret_cast<char *>(&has_right), sizeof(bool));

        if (in.fail())
            return nullptr;

        auto node = std::make_shared<TimeNode>(timestamp, frameidx);
        if (has_left)
            node->left = load_node(in);
        if (has_right)
            node->right = load_node(in);
        return node;
    }
};

#endif // __TIMETREE_HPP__