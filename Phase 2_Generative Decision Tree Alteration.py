from graphviz import Digraph
import random

# Phase 2.1 Programmatic Decision Tree Conversion: Convert the decision tree flow diagram into a programmatic representation.
decision_tree = {
    "Start": {
        "Choose A or B": {
            "A": {
                "Walking": {
                    "Easy": "Straight Corridor",
                    "Moderate": "Multiple Corridor Choices",
                    "Difficult": "Corridor with Obstacles"
                }
            },
            "B": {
                "Stairs": {
                    "Easy": "0-5 Degree Incline",
                    "Moderate": "20-30 Degree Incline",
                    "Difficult": "45-50 Degree Incline"
                }
            }
        }
    },
    "Node A": {
        "Option 1": {
            "Ramps": {
                "Easy": "0-15 Degree Elevation",
                "Moderate": "30-45 Degree Elevation",
                "Difficult": "60 Degree Elevation"
            }
        },
        "Option 2": {
            "Doors": {
                "Easy": "Standard Chronological Doors",
                "Moderate": "Iterative Doors",
                "Difficult": "Chronological Doors with Triggers"
            }
        }
    },
    "Node B": {
        "Choice X or Y": {
            "X": {
                "Walking": {
                    "Easy": "Straight Corridor",
                    "Moderate": "Multiple Corridor Choices",
                    "Difficult": "Corridor with Obstacles"
                }
            },
            "Y": {
                "Ramps": {
                    "Easy": "0-15 Degree Elevation",
                    "Moderate": "30-45 Degree Elevation",
                    "Difficult": "60 Degree Elevation"
                }
            }
        }
    },
    "Node C": {
        "Option 3": {
            "Stairs": {
                "Easy": "0-5 Degree Incline",
                "Moderate": "20-30 Degree Incline",
                "Difficult": "45-50 Degree Incline"
            }
        },
        "Option 4": {
            "Doors": {
                "Easy": "Standard Chronological Doors",
                "Moderate": "Iterative Doors",
                "Difficult": "Chronological Doors with Triggers"
            }
        }
    },
    # ... Continue defining nodes and branches
}

# Spatial Mapping
spatial_mapping = {
    "Node C": "Walking",
    "Node D": "Stairs",
    "Node E": "Ramps",
    "Node F": "Doors",
    # ... Add other nodes
}

# Quantitative Mapping
quantitative_mapping = {
    "Node C": {"Intensity: 0": 0, "Intensity: 1": 1, "Intensity: 2": 2},
    "Node D": {"Intensity: 0": 0, "Intensity: 1": 1, "Intensity: 2": 2},
    "Node E": {"Intensity: 0": 0, "Intensity: 1": 1, "Intensity: 2": 2},
    "Node F": {"Intensity: 0": 0, "Intensity: 1": 1, "Intensity: 2": 2},
    # ... Specify quantitative factors for other nodes
}

dot = Digraph(comment="Spatial Decision Tree")

# Customizing attributes
dot.attr(rankdir="TB")  # Set the layout direction (Top to Bottom)

# Customize node attributes
dot.node_attr.update(style="filled", color="black", shape="point", fontsize="10", fontname="calibri")

# Customize edge attributes
dot.edge_attr.update(fontsize="8", fontcolor="red", arrowhead="tee", arrowsize="0.5")  # Adjust arrowsize as needed

# Phase 2.2" Defining the conditions for generatively altering the decision tree based on spatial translation outputs from phase 1.
def add_nodes_edges(dot, tree, parent_node=None):
    for key, value in tree.items():
        if isinstance(value, dict):
            if parent_node:
                dot.node(f"{parent_node}_{key}", key, shape="box", color="lightblue")
                dot.edge(parent_node, f"{parent_node}_{key}", label=key)
            add_nodes_edges(dot, value, parent_node=f"{parent_node}_{key}" if parent_node else key)
        else:
            dot.node(f"{parent_node}_{key}_{value}", value, shape="box", color="lightgreen")
            dot.edge(parent_node, f"{parent_node}_{key}_{value}", label=key)

# Running counts
instance_count = {"Walking": 0, "Stairs": 0, "Ramps": 0, "Doors": 0}
lowest_qf = {"Walking": float('inf'), "Stairs": float('inf'), "Ramps": float('inf'), "Doors": float('inf')}
highest_qf = {"Walking": float('-inf'), "Stairs": float('-inf'), "Ramps": float('-inf'), "Doors": float('-inf')}

threshold = 10  # Adjust threshold as needed

# Modify decision tree based on phase 1 output
def modify_decision_tree(phase_1_output):
    global instance_count, lowest_qf, highest_qf

    spatial_category, intensity, quantitative_factor = phase_1_output

    # Update counts
    instance_count[spatial_category] += 1

    if quantitative_factor < lowest_qf[spatial_category]:
        lowest_qf[spatial_category] = quantitative_factor

    if quantitative_factor > highest_qf[spatial_category]:
        highest_qf[spatial_category] = quantitative_factor

    # Check if modification is needed
    if instance_count[spatial_category] >= threshold:
        # Modify existing branch
        current_node = random.choice([node for node in decision_tree if spatial_mapping.get(node) == spatial_category])
        current_intensity = random.choice([key for key, value in quantitative_mapping[current_node].items()])
        decision_tree[current_node][f"Intensity: {current_intensity}"] = f"Modified: {spatial_category}_{intensity}_{quantitative_factor}"
    else:
        # Add new branch
        new_node = f"New: {spatial_category}_{intensity}_{quantitative_factor}"
        decision_tree[new_node] = {}

# Phase 2.3 Programmatic Decision Tree Modification: Modify the decision tree based on the output of Phase 1.
# Apply modifications
modify_decision_tree(["Walking", "Moderate", 1])
modify_decision_tree(["Stairs", "Easy", 0])

# Visualize decision tree
add_nodes_edges(dot, decision_tree)
dot.render("complicated_decision_tree_visual_modified", format="png", cleanup=True)

# The final output is an updated decision tree that can be used to modify the spatial plan in phase 3.

