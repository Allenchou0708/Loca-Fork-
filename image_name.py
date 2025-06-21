import os

def list_files_in_folder(folder_path):
    """
    列出指定資料夾下的所有檔案和資料夾名稱。
    """
    try:
        # 使用 os.listdir() 來獲取資料夾內容
        contents = os.listdir(folder_path)
        
        # 過濾只保留檔案（如果需要的話）
        # files = [f for f in contents if os.path.isfile(os.path.join(folder_path, f))]
        
        return contents
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# --- 使用範例 ---
dark_data_folder = "Dark Data" # 替換成你的實際資料夾路徑

file_names = list_files_in_folder(dark_data_folder)

if file_names:
    print(f"Files and folders in '{dark_data_folder}':")
    for name in file_names:
        print(name)