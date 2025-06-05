import networkx as nx
import numpy as np
import argparse
import os

def extract_adjacency_from_graphml(graphml_path, num_nodes, output_path):
    """
    从 GraphML 文件中提取邻接矩阵并保存为 .npy 文件
    """
    # 读取 GraphML 文件
    G = nx.read_graphml(graphml_path)
    
    # 创建邻接矩阵
    adjacency = np.zeros((num_nodes, num_nodes))
    
    # 填充邻接矩阵
    for edge in G.edges():
        # GraphML 可能将节点ID存储为字符串
        i = int(edge[0].replace('n', '')) if edge[0].startswith('n') else int(edge[0])
        j = int(edge[1].replace('n', '')) if edge[1].startswith('n') else int(edge[1])
        adjacency[i, j] = 1
    
    # 保存邻接矩阵
    np.save(output_path, adjacency)
    print(f"Adjacency matrix extracted and saved to: {output_path}")
    
    # 打印一些统计信息
    num_edges = np.sum(adjacency)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {int(num_edges)}")
    print(f"Edge density: {num_edges / (num_nodes * (num_nodes - 1)):.4f}")
    
    return adjacency

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract adjacency matrix from GraphML file')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes')
    parser.add_argument('--graphml', type=str, help='Path to GraphML file (optional)')
    parser.add_argument('--output', type=str, help='Output path for adjacency.npy (optional)')
    
    args = parser.parse_args()
    
    # 默认路径
    if not args.graphml:
        args.graphml = f'{args.num_nodes}/path_graph.graphml'
    if not args.output:
        args.output = f'{args.num_nodes}/adjacency.npy'
    
    # 提取并保存邻接矩阵
    adjacency = extract_adjacency_from_graphml(args.graphml, args.num_nodes, args.output)