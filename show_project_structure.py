# show_project_structure.py
"""Показывает структуру проекта: только .py и .yaml файлы."""

import os
import sys
from pathlib import Path
from datetime import datetime

def get_tree(root_dir):
    """Строит дерево только с .py и .yaml файлами."""
    tree_lines = []
    total_files = 0
    total_dirs = 0
    total_lines = 0
    
    root_dir = os.path.abspath(root_dir)
    project_name = os.path.basename(root_dir)
    
    # Сначала собираем все нужные файлы
    py_yaml_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
            'venv', '.venv', '__pycache__', 'node_modules', '.git',
            '.idea', '.vscode', 'dist', 'build', 'egg-info'
        }]
        for f in files:
            if f.endswith(('.py', '.yaml', '.yml', '.txt')):
                py_yaml_files.append(os.path.join(root, f))
    
    # Группируем по папкам
    from collections import defaultdict
    by_dir = defaultdict(list)
    for f in py_yaml_files:
        rel = os.path.relpath(f, root_dir)
        folder = os.path.dirname(rel)
        by_dir[folder].append((os.path.basename(f), f))
    
    # Рисуем дерево
    tree_lines.append(f"📁 {project_name}/")
    
    sorted_dirs = sorted(by_dir.keys())
    for i, folder in enumerate(sorted_dirs):
        is_last_dir = (i == len(sorted_dirs) - 1)
        dir_prefix = '└── ' if is_last_dir else '├── '
        
        if folder == '':
            files = sorted(by_dir[folder])
            for j, (filename, filepath) in enumerate(files):
                is_last = (j == len(files) - 1)
                prefix = '└── ' if is_last else '├── '
                ext = os.path.splitext(filename)[1]
                icon = '🐍' if ext == '.py' else '⚙️'
                lines = count_lines(filepath)
                total_lines += lines
                tree_lines.append(f"{dir_prefix}{prefix}{icon} {filename} ({lines} строк)")
                total_files += 1
        else:
            tree_lines.append(f"{dir_prefix}📁 {folder}/")
            files = sorted(by_dir[folder])
            for j, (filename, filepath) in enumerate(files):
                is_last = (j == len(files) - 1)
                prefix = '└── ' if is_last else '├── '
                indent = '    ' if is_last_dir else '│   '
                ext = os.path.splitext(filename)[1]
                icon = '🐍' if ext == '.py' else '⚙️'
                lines = count_lines(filepath)
                total_lines += lines
                tree_lines.append(f"{indent}{prefix}{icon} {filename} ({lines} строк)")
                total_files += 1
    
    return tree_lines, total_files, total_lines

def count_lines(filepath):
    """Подсчитывает строки в файле."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return 0

def show_structure(root_dir='.'):
    """Показывает структуру проекта."""
    tree, total_files, total_lines = get_tree(root_dir)
    
    print("=" * 60)
    print(f"📁 СТРУКТУРА ПРОЕКТА ({total_files} файлов)")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    for line in tree:
        print(line)
    
    print("=" * 60)
    print(f"🐍 Python: {sum(1 for l in tree if '🐍' in l)}")
    print(f"⚙️  YAML:  {sum(1 for l in tree if '⚙️' in l)}")
    print(f"📝 Строк: {total_lines:,}")
    print("=" * 60)

def export_to_file(root_dir='.', output_file='project_structure.txt'):
    """Сохраняет в файл."""
    original_stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        show_structure(root_dir)
    sys.stdout = original_stdout
    print(f"✅ {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', nargs='?', default='.')
    parser.add_argument('-e', '--export', action='store_true')
    parser.add_argument('-o', '--output', default='project_structure.txt')
    args = parser.parse_args()
    
    if args.export:
        export_to_file(args.folder, args.output)
    else:
        show_structure(args.folder)