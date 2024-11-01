import os

import tkinter as tk
from tkinter import filedialog

class Menu(tk.Menu):
    def __init__(self, master: tk.Tk, props: dict):
        super().__init__(master)

        self.props = props

        # メニューバー
        menubar = tk.Menu(self.master)

        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(label="ファイルを開く", command=self.open_filedialog)
        menu_file.add_command(label="名前を付けて保存", command=lambda: print("名前を付けて保存"))

        menubar.add_cascade(label="ファイル", menu=menu_file)

        # メニューバーの設定
        self.master.config(menu=menubar)
    
    def open_filedialog(self):
        # ファイルフィルタ
        file_type = [("wavファイル", "*.wav"), ("", "*")]
        # 最初に開くフォルダ
        initial_directory_path = os.path.abspath(os.path.dirname(__file__))

        file_name = filedialog.askopenfilename(
            filetypes=file_type, initialdir=initial_directory_path
        )
        
        # ファイルを開いたら, イベントを発生させる
        if file_name != "":
            self.props["open_file_name"] = file_name
            self.master.event_generate("<<OpenFile>>")