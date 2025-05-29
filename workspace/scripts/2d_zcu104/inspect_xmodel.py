import xir
xmodel_path = "/workspace/outputs2d_zcu104/compile_out/vxm2d.xmodel"
graph = xir.Graph.deserialize(xmodel_path)
root = graph.get_root_subgraph()
children = root.get_children()  

print(f"Found {len(children)} total subgraphs:\n")
for idx, sg in enumerate(children):
    name= sg.get_name()
    kind= sg.get_attr("device") if sg.has_attr("device") else sg.get_type()
    print(f"[{idx:02d}]{name:40s} type={kind}")
