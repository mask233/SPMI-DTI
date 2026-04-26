#!/bin/bash

# 如果log文件夹不存在，则创建
if [ ! -d "log" ]; then
    mkdir log
fi

# 获取当前时间，格式为YYYY-MM-DD_HH-MM-SS
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
comment="MIF-DTI"
file_name="log/$current_time-$comment.log"

# 运行nohup命令，输出日志到log目录下，文件名为当前时间
echo $comment > "$file_name"
nohup python -u main.py BIOSNAP -g 0    >> "$file_name" 2>&1 & 

# 实时显示日志文件内容
tail -f "$file_name"
