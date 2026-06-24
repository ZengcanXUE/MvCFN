import os
from collections import defaultdict


def split_knowledge_graph(input_filename="test.txt", output_dir="output_data"):
    """
    读取知识图谱三元组文件，并根据关系类型拆分为多个文件。

    :param input_filename: 包含原始知识图谱数据的输入文件名。
    :param output_dir: 存放输出文件的目录名称。
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_filename):
        print(f"错误: 输入文件 '{input_filename}' 未找到。请确保文件存在。")
        return

    # 2. 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录已准备: {output_dir}")
    except Exception as e:
        print(f"创建目录时出错: {e}")
        return

    # 3. 读取并按关系分类数据
    # 使用 defaultdict 自动初始化一个列表给新的关系
    relationship_data = defaultdict(list)
    total_triples = 0

    print(f"开始处理文件: {input_filename}")

    with open(input_filename, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            # 清理行首尾的空白字符
            clean_line = line.strip()

            # 忽略空行
            if not clean_line:
                continue

            # 尝试按制表符分割三元组
            parts = clean_line.split('\t')

            if len(parts) == 3:
                # 假设格式是：<实体1>\t<关系>\t<实体2>
                entity1, relation, entity2 = parts

                # 移除关系名称中的下划线，作为文件名 (可选, 但更美观)
                # 例如: _hypernym -> hypernym
                relation_name = relation.strip().lstrip('_')

                # 存储原始的三元组行（使用制表符连接，方便后续写入）
                relationship_data[relation_name].append(clean_line)
                total_triples += 1
            else:
                print(f"警告: 第 {line_number} 行格式不正确，跳过: '{clean_line}'")

    print(f"处理完成。总计找到 {total_triples} 条有效三元组。")
    print(f"发现 {len(relationship_data)} 种不同的关系类型。")

    # 4. 将每种关系写入单独的文件
    for relation_name, triples in relationship_data.items():
        output_filename = os.path.join(output_dir, f"{relation_name}.txt")

        # 写入文件，每行一个三元组
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            for triple in triples:
                f_out.write(triple + '\n')

        print(f"  - 关系 '{relation_name}' 写入文件: {output_filename} ({len(triples)} 条)")

    print("\n所有关系文件已成功创建。")
    print(f"您可以在 '{output_dir}' 目录中找到它们。")


# --- 运行代码 ---
if __name__ == "__main__":
    # 请将 input_filename 设置为您上传的文件的实际名称，如果与上面的一致，则无需修改。
    # 默认使用 'test.txt' 作为输入文件名
    split_knowledge_graph(input_filename="test.txt")