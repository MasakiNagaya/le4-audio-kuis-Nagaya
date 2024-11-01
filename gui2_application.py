import tkinter as tk

from gui2_audio import Audio
from gui2_menu import Menu

class Application(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.title = "Tkinter Application"

        master.geometry("800x600")
        master.title(self.title)

        # クラス間で共通で使われる変数
        props: dict = {}

        Menu(self.master, props)

        self.mainframe = tk.Frame(self)
        self.mainframe.pack()

        self.audio = Audio(self.mainframe, props)