import sys
import os
import random
import chess
import chess.svg
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit
from PyQt5.QtSvg import QSvgWidget
from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
import json
import re

# ---- LlamaIndex imports ----
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage,set_global_service_context, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from datetime import datetime
RULES_FILE = "data"
STORAGE_DIR = "storage"


llm_model_path = "llama-3.2-3b-instruct-q4_k_m.gguf"
llm = Llama(model_path=llm_model_path, max_tokens=128)
llm = LlamaCPP(model_path=llm_model_path, temperature=0.2, max_new_tokens=256)
#service_context = ServiceContext.from_defaults(llm=llm)
#set_global_service_context(service_context)
Settings.llm = llm


def load_or_build_index():
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

    #if Path(STORAGE_DIR).exists():
       # storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
       # index = load_index_from_storage(storage_context,embed_model=embed_model, llm=llm )
    #else:
        docs = SimpleDirectoryReader(RULES_FILE).load_data()
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model, llm=llm)
        index.storage_context.persist(persist_dir=STORAGE_DIR)

    # 建立 query engine（非常重要）
        query_engine = index.as_query_engine()
        return index, query_engine

index, query_engine = load_or_build_index()
# ------------------ Chess Agent ------------------
class ChessAgent:
    def __init__(self, use_stockfish=True):
        self.board = chess.Board()
        self.difficulty = None
        self.move_history = []  # [(player_move, ai_move)]
        self.stockfish_path = r"C:\Users\Administrator\Desktop\AIchess\stockfish\stockfish-windows-x86-64-avx2.exe"
        #self.history_file = f"chess_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        if use_stockfish and Path(self.stockfish_path).exists():
            from stockfish import Stockfish
            self.engine = Stockfish(path=self.stockfish_path)
        else:
            self.engine = None
        print(self.engine)
    def set_difficulty(self, level):
        self.difficulty = max(1, min(level, 10))
        if self.engine:
            self.engine.set_skill_level(self.difficulty)

    def apply_move(self, move_uci):
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except:
            return False

    def engine_bestmove(self):
        if self.engine:
            self.engine.set_fen_position(self.board.fen())
            return self.engine.get_best_move()
        return None

    def player_bestmove(self):
        """詢問：如果我是現在這一方，最佳棋步是什麼"""
        if not self.engine:
            return None
        self.engine.set_fen_position(self.board.fen())
        return self.engine.get_best_move()

    def undo(self):
        """悔棋：回退玩家 + AI 各一步"""
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            if self.move_history:
                self.move_history.pop()
            return True
        return False

    def render_board_svg(self):
        return chess.svg.board(self.board)
# ------------------ GUI ------------------
class ChessGUI(QWidget):
    def __init__(self, agent, query_engine):
        super().__init__()
        self.agent = agent
        self.query_engine = query_engine
        self.difficulty_set = False
        self.history_file = f"chess_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.setWindowTitle("Chess JSON Agent with LlamaIndex")
        self.setGeometry(100, 100, 500, 600)
        layout = QVBoxLayout()

        self.svg_widget = QSvgWidget()
        layout.addWidget(self.svg_widget)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(50)
        layout.addWidget(self.input_box)

        button = QPushButton("送出")
        layout.addWidget(button)
        button.clicked.connect(self.on_input)

        self.setLayout(layout)
        self.update_board()

    def update_board(self):
        svg = self.agent.render_board_svg().encode()
        self.svg_widget.load(svg)

    def save_history(self, player_move, ai_move):
        with open(self.history_file, "a", encoding="utf-8") as f:
            turn = len(self.agent.move_history)
            f.write(f"{turn}. Player: {player_move}\n")
            f.write(f"   AI: {ai_move}\n")

    # ==== LLM + Index查詢 ====
    def ask_llm(self, user_input):

        # === 修正：新版用 as_query_engine ====
        query_res = self.query_engine.query(user_input)
        context_text = str(query_res)
        print("Index Context:", context_text)
        # === 用 index 內容當作 system prompt ===
        #prompt = f"<s>[INST]{context_text}\n玩家輸入:{user_input}[/INST]"
        #out = llm.complete(prompt, max_tokens=128)
        #print("LLM Raw Output:", out)
        #text = out["choices"][0]["text"]
        #text = out.text
        #print("LLM Processed Output:", text)
        return context_text

    # ==== 解析 JSON ====
    def parse_json(self, text):
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except:
            return None

    # ==== 主流程 ====
    # ==== 主流程 ====
    def on_input(self):
        if self.input_box.toPlainText().strip() == "":
            return
        user = self.input_box.toPlainText().strip()
        self.input_box.clear()
        self.chat_display.append(f"你> {user}")

        llm_output = self.ask_llm(user)
        data = self.parse_json(llm_output)

        if not data:
            self.chat_display.append("⚠️ 無法解析 JSON")
            return

        action = data.get("action")
        player_move = data.get("player_move")
        level = data.get("level")

        # --- 設定難度 ---
        if not self.difficulty_set:
            if action == "set_difficulty":
                self.agent.set_difficulty(int(level))
                self.difficulty_set = True
                self.chat_display.append(f"難度設定為 {level}")
            else:
                self.chat_display.append("⚠️ Must set difficulty first")
            return

        # --- 詢問我方最佳棋 ---
        if action == "player_bestmove":
            best = self.agent.player_bestmove()
            self.chat_display.append(f"📌 建議你下：{best}")
            return

        # --- 悔棋 ---
        if action == "undo":
            if self.agent.undo():
                self.chat_display.append("↩️ 悔棋成功")
                self.update_board()
            else:
                self.chat_display.append("⚠️ 無法悔棋")
            return

        # --- 玩家下棋 ---
        if action == "engine_bestmove":
            if not self.agent.apply_move(player_move):
                self.chat_display.append(f"⚠️ 非法棋步: {player_move}")
                return

            ai = self.agent.engine_bestmove()
            self.agent.apply_move(ai)

            self.agent.move_history.append((player_move, ai))
            self.save_history(player_move, ai)

            self.chat_display.append(f"你下: {player_move}")
            self.chat_display.append(f"AI 下: {ai}")

        self.update_board()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ChessGUI(ChessAgent(), query_engine)
    gui.show()
    sys.exit(app.exec_())