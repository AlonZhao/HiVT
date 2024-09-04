from argparse import ArgumentParser
# 终端调用方式
# (base) alon@AlonLegion:~/Learning/HiVT$ python argp_test.py alon --age 13
# 你好，alon！
# 你的年龄是 13 岁。

# 创建 ArgumentParser 对象
parser = ArgumentParser(description="一个简单的命令行工具示例")
# 添加位置参数必须提供
parser.add_argument('name', type=str, help='用户的名字')
# 添加可选参数
parser.add_argument('--age', type=int, help='用户的年龄')


group = parser.add_argument_group('HiVT')
group.add_argument('--learning-rate', type=float, default=0.01, help='学习率')
group.add_argument('--num-layers', type=int, default=2, help='网络层数')
# 解析命令行参数解析命令行中提供的参数，并将解析后的参数存储在 args 对象
args = parser.parse_args()


# 使用参数
print(f"你好，{args.name}！")
if args.age:
    print(f"你的年龄是 {args.age} 岁。")

# 假设 args 是通过 parser.parse_args() 解析得到的 Namespace 对象

# for arg_name, arg_value in vars(args).items():
#     print(f"{arg_name} {arg_value}")

print(vars(args))#{'name': 'alon', 'age': 13, 'learning_rate': 0.01, 'num_layers': 2}

print(**vars(args))