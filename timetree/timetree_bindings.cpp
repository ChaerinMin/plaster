#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>

#include "TimeTree.hpp"

namespace py = pybind11;

// Helper to return a lightweight Python dict for a node (or None)
static py::object node_to_dict(const std::shared_ptr<TimeNode>& n) {
    if (!n) return py::none();
    py::dict d;
    d["timestamp"] = n->timestamp;
    d["arbitrary_node_info"] = n->arbitrary_node_info;
    d["height"] = n->height;
    // Expose child presence only (avoid deep recursion by default)
    d["has_left"] = static_cast<bool>(n->left);
    d["has_right"] = static_cast<bool>(n->right);
    return std::move(d);
}

PYBIND11_MODULE(timetree, m) {
    m.doc() = "Python bindings for TimeTree (AVL-based timestamp index)";

    py::class_<TimeNode, std::shared_ptr<TimeNode>>(m, "TimeNode")
        .def_property_readonly("timestamp", [](const TimeNode& self){ return self.timestamp; })
        .def_property_readonly("arbitrary_node_info", [](const TimeNode& self){ return self.arbitrary_node_info; })
        .def_property_readonly("height", [](const TimeNode& self){ return self.height; })
        .def_property_readonly("left", [](const TimeNode& self){ return self.left; })
        .def_property_readonly("right", [](const TimeNode& self){ return self.right; })
        ;

    py::class_<TimeTree, std::shared_ptr<TimeTree>>(m, "TimeTree")
        .def(py::init<>(),
             "Construct an empty TimeTree (no file loaded).")
       .def(py::init<const std::string&>(),
           py::arg("filename"),
           "Construct a TimeTree from a metadata file (lines: prefix_timestamp_frame).")
        .def_static("load", &TimeTree::load, py::arg("binfile"),
                    "Load a serialized TimeTree from a binary file.")
        .def("save", &TimeTree::save, py::arg("binfile"),
             "Serialize the TimeTree to a binary file.")
        .def("get_raw", &TimeTree::get, py::arg("timestamp"), py::arg("threshold") = 1000,
             py::return_value_policy::reference, "Find closest node (returns TimeNode or None).")
        .def("get", [](std::shared_ptr<TimeTree> self, std::int64_t ts, std::int64_t threshold){
            return node_to_dict(self->get(ts, threshold));
        }, py::arg("timestamp"), py::arg("threshold") = 1000,
        "Find closest node; returns dict with timestamp/arb_info or None.")
       // Expose public build/append helpers directly
       .def("buildAVLTree", &TimeTree::buildAVLTree,
           py::arg("timestamp_filepath"), py::arg("root") = nullptr,
           py::return_value_policy::reference,
           "Build an AVL tree from a timestamp file (optionally from an existing root); returns the root.")
       .def("appendAVLTree", &TimeTree::appendAVLTree,
           py::arg("timestamp_filepath"),
           py::return_value_policy::reference,
           "Append entries from a timestamp file into the existing AVL tree; returns the root.")
        .def_property_readonly("root", [](TimeTree& self){ return self.m_root; })
        .def("height", [](TimeTree& self){ return self.getTreeDepth(self.m_root); })
        .def("nodes", [](TimeTree& self){ return self.getTotalNodes(self.m_root); })
        .def("leaves", [](TimeTree& self){ return self.countLeafNodes(self.m_root); })
        ;
}
