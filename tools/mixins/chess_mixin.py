"""
tools/mixins/chess_mixin.py — Chess board detection and mouse control mixin.

Detects a chess board on screen by identifying the characteristic square colours,
finds piece positions via template matching, determines the player's colour from
the lower portion of the board, and performs natural mouse-driven moves.

Board colours:
  Purple / dark squares : #8476ba
  White  / light squares: #f0f1f0

Piece image directory: tools/mixins/media/chess_pieces/
  br bn bb bk bq bp  (black pieces)
  wr wn wb wk wq wp  (white pieces)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
import urllib.parse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.base_mixin import ToolMixin, tool_function, fire_and_forget

# Board background colours (BGR order for OpenCV).
# All known square colours from various chess sites/themes — no
# distinction between light and dark squares.
_BG_BGRS = [
    np.array([0x46, 0x57, 0xBB], dtype=np.int32),  # #bb5746
    np.array([0xC3, 0xDB, 0xF5], dtype=np.int32),  # #f5dbc3
    np.array([0x6C, 0xA8, 0xD9], dtype=np.int32),  # #d9a86c
    np.array([0xAA, 0xE9, 0xF6], dtype=np.int32),  # #f6e9aa
    np.array([0xA7, 0xBC, 0xD2], dtype=np.int32),  # #d2bca7
]
_COLOUR_TOL = 12  # per-channel tolerance for colour matching

# Piece template directory (relative to this file)
_PIECE_DIR = Path(__file__).parent / "media" / "chess_pieces"

# Piece codes used in template filenames
_PIECE_CODES = [
    "br", "bn", "bb", "bq", "bk", "bp",  # black pieces
    "wr", "wn", "wb", "wq", "wk", "wp",  # white pieces
]

# Map piece code → FEN character
_CODE_TO_FEN = {
    "bp": "p", "bn": "n", "bb": "b", "br": "r", "bq": "q", "bk": "k",
    "wp": "P", "wn": "N", "wb": "B", "wr": "R", "wq": "Q", "wk": "K",
}

# Column labels
_FILES = "abcdefgh"


class ChessMixin(ToolMixin):
    """Provides chess board reading and move-execution tools."""

    MIXIN_NAME = "chess"

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._piece_templates: dict[str, np.ndarray] = {}
        self._piece_masks: dict[str, np.ndarray | None] = {}
        self._load_piece_templates()
        # Move history tracking
        self._last_grid: list[list[str | None]] | None = None
        self._last_player: str | None = None
        self._move_history: list[str] = []   # e.g. ["1. e2-e4", "1... e7-e5", ...]
        self._move_number: int = 1

    # ------------------------------------------------------------------
    # Move detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sq_name(row: int, col: int) -> str:
        """Convert (row, col) in the normalised grid to algebraic notation."""
        return _FILES[col] + str(8 - row)

    def _detect_move(
        self,
        old: list[list[str | None]],
        new: list[list[str | None]],
        moving_colour: str,  # "w" or "b"
    ) -> str | None:
        """
        Compare two normalised 8×8 grids and return a human-readable move string,
        e.g. ``"Nb1-c3"`` or ``"e2-e4"`` or ``"O-O"``.
        Returns None if no single move can be detected.
        """
        removed: list[tuple[int, int, str]] = []   # (row, col, piece_code)
        appeared: list[tuple[int, int, str]] = []  # (row, col, piece_code)

        for r in range(8):
            for c in range(8):
                o, n = old[r][c], new[r][c]
                if o == n:
                    continue
                if o and not n:
                    removed.append((r, c, o))
                elif not o and n:
                    appeared.append((r, c, n))
                elif o and n and o != n:
                    # Square changed occupant (capture in place or promotion)
                    removed.append((r, c, o))
                    appeared.append((r, c, n))

        if not removed or not appeared:
            return None

        # --- Castling detection: king + rook both moved ---
        moving_pieces = [p for _, _, p in removed if p[0] == moving_colour]
        if len(moving_pieces) == 2 and any(p[1] == "k" for p in moving_pieces):
            king_dest = next((c for r, c, p in appeared if p[0] == moving_colour and p[1] == "k"), None)
            if king_dest is not None:
                return "O-O" if king_dest == 6 else "O-O-O"

        # --- Normal move: identify the piece that moved ---
        # The moving piece is the one that disappeared from a square of the moving colour
        from_entries = [(r, c, p) for r, c, p in removed if p[0] == moving_colour]
        if not from_entries:
            return None
        fr, fc, piece_code = from_entries[0]
        from_sq = self._sq_name(fr, fc)

        # Destination: where the same colour piece appeared (could be promoted)
        to_entries = [(r, c, p) for r, c, p in appeared if p[0] == moving_colour]
        if not to_entries:
            return None
        tr, tc, dest_code = to_entries[0]
        to_sq = self._sq_name(tr, tc)

        # Was there a capture? (enemy piece disappeared or square was occupied)
        capture_sq = to_sq
        was_capture = (old[tr][tc] is not None) and (old[tr][tc][0] != moving_colour)  # type: ignore[index]
        # En passant: pawn moved diagonally to empty square — captured pawn removed elsewhere
        if not was_capture and piece_code[1] == "p" and fc != tc:
            was_capture = True

        piece_type = piece_code[1]  # 'p','n','b','r','q','k'
        piece_letter = {"n": "N", "b": "B", "r": "R", "q": "Q", "k": "K"}.get(piece_type, "")

        # Format
        if piece_type == "p":
            if was_capture:
                notation = f"{_FILES[fc]}x{to_sq}"
            else:
                notation = f"{from_sq}-{to_sq}"
            # Promotion
            if dest_code[1] != "p":
                notation += f"={dest_code[1].upper()}"
        else:
            sep = "x" if was_capture else "-"
            notation = f"{piece_letter}{from_sq}{sep}{to_sq}"

        return notation

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_piece_templates(self) -> None:
        """Load piece PNGs from the media directory.

        Templates are loaded with alpha so we can mask out the transparent
        background during matching — critical for dark pieces on dark squares.
        """
        for code in _PIECE_CODES:
            path = _PIECE_DIR / f"{code}.png"
            if path.exists():
                raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    continue
                if raw.shape[2] == 4:  # has alpha
                    bgr = raw[:, :, :3]
                    alpha = raw[:, :, 3]
                    # Binary mask: opaque pixels → 255
                    mask = (alpha > 128).astype(np.uint8) * 255
                    self._piece_templates[code] = bgr
                    self._piece_masks[code] = mask
                else:
                    self._piece_templates[code] = raw
                    self._piece_masks[code] = None  # no mask
            else:
                self.logger.warning("Missing piece template: %s", path)

    # ------------------------------------------------------------------
    # Screenshot (XDG Desktop Portal — Wayland safe)
    # ------------------------------------------------------------------

    def _screenshot_bgr(self) -> np.ndarray:
        """
        Capture the full screen via the XDG Desktop Portal Screenshot API
        (org.freedesktop.portal.Screenshot).  Works on GNOME Wayland without
        needing compositor-specific tools.

        Flow:
          1. Open a session-bus connection (jeepney, pure Python).
          2. Add a match rule so we receive the Response signal.
          3. Call Screenshot("", {interactive: false}).
          4. The method returns a request object-path immediately.
          5. Loop-receive until we get the Response signal on that path.
          6. Parse the file:// URI from the results dict, read with OpenCV.
        """
        from jeepney import DBusAddress, MessageType, new_method_call
        from jeepney.bus_messages import MatchRule, message_bus, HeaderFields
        from jeepney.io.blocking import open_dbus_connection

        portal_addr = DBusAddress(
            "/org/freedesktop/portal/desktop",
            bus_name="org.freedesktop.portal.Desktop",
            interface="org.freedesktop.portal.Screenshot",
        )

        with open_dbus_connection(bus="SESSION") as conn:
            match = MatchRule(
                type="signal",
                interface="org.freedesktop.portal.Request",
                member="Response",
            )
            conn.send_and_get_reply(message_bus.AddMatch(match))

            call_msg = new_method_call(
                portal_addr,
                "Screenshot",
                "sa{sv}",
                ("", {"interactive": ("b", False)}),
            )
            reply = conn.send_and_get_reply(call_msg)
            request_path: str = reply.body[0]

            response_code: int = 1
            results: dict = {}
            for _ in range(200):  # up to ~20 s
                try:
                    incoming = conn.receive(timeout=0.1)
                except Exception:
                    continue
                if incoming.header.message_type != MessageType.signal:
                    continue
                sig_path = incoming.header.fields.get(HeaderFields.path, "")
                if sig_path != request_path:
                    continue
                response_code, results = incoming.body
                break

        if response_code != 0:
            raise RuntimeError(
                f"Portal screenshot failed (response code {response_code})."
            )

        uri: str = results.get("uri", ("", ""))[1]
        if not uri:
            raise RuntimeError("Portal screenshot returned no URI.")

        file_path = urllib.parse.urlparse(uri).path
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        try:
            os.unlink(file_path)
        except OSError:
            pass

        if img is None:
            raise RuntimeError(f"Could not read screenshot from {file_path!r}.")
        return img

    # ------------------------------------------------------------------
    # Board detection
    # ------------------------------------------------------------------

    def _detect_board(self, img: np.ndarray) -> dict | None:
        """
        Find the chess board region in *img* (BGR).

        Returns a dict with keys ``x, y, w, h, sq_size`` or ``None``
        if no board-shaped region is found.
        """
        img32 = img.astype(np.int32)
        board_mask = np.zeros(img32.shape[:2], dtype=bool)
        for bgr in _BG_BGRS:
            board_mask |= np.all(np.abs(img32 - bgr) <= _COLOUR_TOL, axis=2)
        combined = board_mask.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best, best_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) else 0
            if area > best_area and aspect >= 0.70:
                best_area = area
                sq = min(cw, ch) // 8
                best = {
                    "x": x, "y": y,
                    "w": sq * 8, "h": sq * 8,
                    "sq_size": sq,
                }
        return best

    # ------------------------------------------------------------------
    # Piece recognition
    # ------------------------------------------------------------------

    @staticmethod
    def _foreground_mask(cell: np.ndarray) -> np.ndarray:
        """
        Return a single-channel uint8 mask (0/255) where 255 = foreground
        (pixel does NOT match any known board background colour).
        """
        c32 = cell.astype(np.int32)
        bg = np.zeros(c32.shape[:2], dtype=bool)
        for bgr in _BG_BGRS:
            bg |= np.all(np.abs(c32 - bgr) <= _COLOUR_TOL + 5, axis=2)
        return (~bg).astype(np.uint8) * 255

    def _identify_pieces(
        self, img: np.ndarray, board: dict
    ) -> list[list[str | None]]:
        """
        Return an 8×8 grid (row-major, rank 8 at index 0) where each cell
        is a piece code like ``"wr"`` or ``None`` for empty.

        1. For each cell, build a foreground mask (remove all board bg colours).
        2. Match the binary foreground silhouette against template silhouettes
           to determine piece TYPE (pawn, knight, bishop, rook, queen, king).
        3. Determine piece COLOUR from the mean brightness of foreground pixels.
        """
        grid: list[list[str | None]] = [[None] * 8 for _ in range(8)]
        sq = board["sq_size"]
        if not self._piece_templates:
            self.logger.error("No piece templates loaded — cannot identify pieces.")
            return grid

        # Build one binary silhouette per piece TYPE from the alpha masks.
        # Shape is identical between black/white, so we use the white set.
        silhouettes: dict[str, np.ndarray] = {}
        for code in ("wp", "wn", "wb", "wr", "wq", "wk"):
            raw_mask = self._piece_masks.get(code)
            if raw_mask is None:
                continue
            sil = cv2.resize(raw_mask, (sq, sq), interpolation=cv2.INTER_NEAREST)
            piece_type = code[1]  # 'p', 'n', 'b', 'r', 'q', 'k'
            silhouettes[piece_type] = sil

        for row in range(8):
            for col in range(8):
                x0 = board["x"] + col * sq
                y0 = board["y"] + row * sq
                cell = img[y0: y0 + sq, x0: x0 + sq]
                if cell.shape[0] != sq or cell.shape[1] != sq:
                    continue

                # --- Foreground mask: remove all board bg colours ---
                fg_mask = self._foreground_mask(cell)
                fg_ratio = fg_mask.sum() / (255.0 * fg_mask.size)

                # Skip empty squares (< 5% foreground)
                if fg_ratio < 0.05:
                    continue

                # --- Step 1: match piece SHAPE via binary silhouettes ---
                best_type: str | None = None
                best_val = -1.0
                for ptype, sil in silhouettes.items():
                    res = cv2.matchTemplate(
                        fg_mask, sil, cv2.TM_CCOEFF_NORMED
                    )
                    val = float(res.max())
                    if val > best_val:
                        best_val = val
                        best_type = ptype

                if best_val < 0.40 or best_type is None:
                    continue

                # --- Step 2: determine piece COLOUR from mean BGR ---
                # Use a generous tolerance to strip ALL background pixels
                c32 = cell.astype(np.int32)
                bg = np.zeros(c32.shape[:2], dtype=bool)
                for bgr in _BG_BGRS:
                    bg |= np.all(np.abs(c32 - bgr) <= 30, axis=2)
                fg_pixels = cell[~bg]
                if len(fg_pixels) == 0:
                    continue
                mean_bgr = fg_pixels.mean(axis=0)  # shape (3,)
                # Black pieces template mean BGR ≈ (75, 77, 80)
                # White pieces template mean BGR ≈ (198, 199, 199)
                # Midpoint ≈ 137
                avg = float(mean_bgr.mean())
                colour = "w" if avg > 137 else "b"

                grid[row][col] = colour + best_type

        return grid

    # ------------------------------------------------------------------
    # Orientation detection  (is the player white or black?)
    # ------------------------------------------------------------------

    def _detect_player_colour(self, grid: list[list[str | None]]) -> str:
        """
        Heuristic: if the bottom two ranks (rows 6-7) contain mostly white
        pieces the player is white; otherwise black.
        """
        white_bottom = 0
        black_bottom = 0
        for row in (6, 7):
            for cell in grid[row]:
                if cell and cell.startswith("w"):
                    white_bottom += 1
                elif cell and cell.startswith("b"):
                    black_bottom += 1
        return "white" if white_bottom >= black_bottom else "black"

    # ------------------------------------------------------------------
    # FEN generation
    # ------------------------------------------------------------------

    def _grid_to_fen(
        self, grid: list[list[str | None]], player: str
    ) -> str:
        """
        Convert the 8×8 grid to a FEN position string.

        *player* is ``"white"`` or ``"black"`` and determines whose
        turn we assume it to be (``w`` / ``b``).
        """
        rows: list[str] = []
        for rank_row in grid:
            empty = 0
            rank_str = ""
            for cell in rank_row:
                if cell is None:
                    empty += 1
                else:
                    if empty:
                        rank_str += str(empty)
                        empty = 0
                    rank_str += _CODE_TO_FEN.get(cell, "?")
            if empty:
                rank_str += str(empty)
            rows.append(rank_str)

        turn = "w" if player == "white" else "b"
        return " ".join(["/".join(rows), turn, "KQkq", "-", "0", "1"])

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _square_to_pixel(
        self, board: dict, square: str, player: str
    ) -> tuple[int, int]:
        """
        Convert an algebraic square (e.g. ``"e2"``) to screen pixel
        coordinates (centre of the square), accounting for board orientation.
        """
        file_idx = _FILES.index(square[0])
        rank_idx = int(square[1]) - 1  # 0-based from rank 1

        if player == "white":
            col = file_idx
            row = 7 - rank_idx
        else:
            col = 7 - file_idx
            row = rank_idx

        sq = board["sq_size"]
        px = board["x"] + col * sq + sq // 2
        py = board["y"] + row * sq + sq // 2
        return px, py

    # ------------------------------------------------------------------
    # Mouse move helper (natural-looking Bézier drag)
    # ------------------------------------------------------------------

    @staticmethod
    def _bezier_points(
        start: tuple[int, int],
        end: tuple[int, int],
        steps: int = 30,
    ) -> list[tuple[int, int]]:
        """Generate a smooth Bézier curve between two points."""
        sx, sy = start
        ex, ey = end
        cx = (sx + ex) // 2 + random.randint(-40, 40)
        cy = (sy + ey) // 2 + random.randint(-40, 40)
        points: list[tuple[int, int]] = []
        for i in range(steps + 1):
            t = i / steps
            x = int((1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t ** 2 * ex)
            y = int((1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t ** 2 * ey)
            points.append((x, y))
        return points

    async def _mouse_move(self, from_sq: str, to_sq: str, board: dict, player: str) -> None:
        """
        Perform a click-drag move from *from_sq* to *to_sq* using pynput,
        with a natural Bézier curve.
        """
        from pynput.mouse import Button, Controller

        mouse = Controller()
        start = self._square_to_pixel(board, from_sq, player)
        end = self._square_to_pixel(board, to_sq, player)

        mouse.position = start
        await asyncio.sleep(0.05)
        mouse.press(Button.left)
        await asyncio.sleep(0.03)

        for pt in self._bezier_points(start, end):
            mouse.position = pt
            await asyncio.sleep(random.uniform(0.004, 0.012))

        mouse.position = end
        await asyncio.sleep(0.03)
        mouse.release(Button.left)

    # ==================================================================
    # Tool functions (exposed to Gemini)
    # ==================================================================

    @tool_function(
        description=(
            "Take a screenshot, detect the chess board, identify all pieces, "
            "and return the current position as a FEN string, a board diagram, "
            "and the move history in algebraic notation."
        ),
    )
    async def read_chess_board(self) -> str:
        """Screenshot → detect board → identify pieces → return FEN + move history."""
        img = await asyncio.to_thread(self._screenshot_bgr)
        board = self._detect_board(img)
        if board is None:
            return "ERROR: No chess board detected on screen."

        grid = self._identify_pieces(img, board)
        player = self._detect_player_colour(grid)

        # Normalise grid so grid[0] = rank 8 (standard FEN order, white at bottom).
        if player == "black":
            norm_grid = [row[::-1] for row in grid[::-1]]
        else:
            norm_grid = [row[:] for row in grid]

        # --- Detect move since last call ---
        if self._last_grid is not None:
            moving_colour = "w" if player == "white" else "b"
            # Flip perspective so grids are comparable (both white-at-bottom)
            move_str = self._detect_move(self._last_grid, norm_grid, moving_colour)
            if move_str:
                if moving_colour == "w":
                    self._move_history.append(f"{self._move_number}. {move_str}")
                else:
                    self._move_history.append(f"{self._move_number}... {move_str}")
                    self._move_number += 1

        self._last_grid = norm_grid
        self._last_player = player

        fen = self._grid_to_fen(norm_grid, player)

        # Board diagram
        diagram_lines: list[str] = []
        for rank_idx, rank_row in enumerate(norm_grid):
            rank_num = 8 - rank_idx
            cells = []
            for cell in rank_row:
                if cell is None:
                    cells.append(".")
                else:
                    cells.append(_CODE_TO_FEN.get(cell, "?"))
            diagram_lines.append(f"  {rank_num}  {' '.join(cells)}")
        diagram_lines.append("     a b c d e f g h")
        diagram = "\n".join(diagram_lines)

        # Move history (last 30 entries to stay concise)
        if self._move_history:
            history_str = " ".join(self._move_history[-30:])
        else:
            history_str = "(no moves recorded yet)"

        return (
            f"FEN: {fen}\n"
            f"You're playing: {player}\n\n"
            f"Board (. = empty, uppercase = white, lowercase = black):\n"
            f"{diagram}\n\n"
            f"Move history: {history_str}\n\n"
            f"Board origin: ({board['x']}, {board['y']})  "
            f"Square size: {board['sq_size']}px"
        )

    @tool_function(
        description=(
            "Reset the move history and board state tracker. "
            "Call this at the start of a new game so the move log starts fresh."
        ),
    )
    async def reset_chess_history(self) -> str:
        """Clear stored board state and move history."""
        self._last_grid = None
        self._last_player = None
        self._move_history = []
        self._move_number = 1
        return "Chess move history reset."

    @tool_function(
        description=(
            "Execute a chess move on the screen board by dragging the piece "
            "from one square to another. Provide the move in algebraic "
            "coordinate notation, e.g. from_square='e2', to_square='e4'."
        ),
        parameter_descriptions={
            "from_square": "Source square in algebraic notation (e.g. 'e2').",
            "to_square": "Destination square in algebraic notation (e.g. 'e4').",
        },
    )
    @fire_and_forget
    async def make_chess_move(self, from_square: str, to_square: str) -> str:
        """Detect board, then drag the piece from *from_square* to *to_square*."""
        from_square = from_square.lower().strip()
        to_square = to_square.lower().strip()

        for sq in (from_square, to_square):
            if len(sq) != 2 or sq[0] not in _FILES or sq[1] not in "12345678":
                return f"ERROR: Invalid square '{sq}'. Use algebraic notation (a1-h8)."

        img = await asyncio.to_thread(self._screenshot_bgr)
        board = self._detect_board(img)
        if board is None:
            return "ERROR: No chess board detected on screen."

        grid = self._identify_pieces(img, board)
        player = self._detect_player_colour(grid)

        await self._mouse_move(from_square, to_square, board, player)

        return f"Attempted {from_square} → {to_square} (playing as {player}). Verify with read_chess_board if needed."

