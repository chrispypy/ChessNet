import chess
import numpy as np

class Encoder:
    def __init__(self):
        self.policy_size = 8*8*(7*8) #wie alpha zero, aber nur queen-like moves
        self.offset_dirs = [ # Alle richtungen in die sich eine Dame bewegen kann
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ] 
        self.create_lut()
        self._create_augmentation_luts()

    def create_lut(self):
        self.encode_lut = {}
        self.decode_lut: list[chess.Move | None] = [None]*self.policy_size
        for x_from in range(8):
            for y_from in range(8):
                from_square = chess.square(x_from, y_from)
                for dir_idx, offset in enumerate(self.offset_dirs):
                    x_to = x_from
                    y_to = y_from
                    for distance in range(7):
                        x_to += offset[0]
                        y_to += offset[1]
                        if not (0 <= x_to < 8 and 0 <= y_to < 8):
                            break
                        to_square = chess.square(x_to, y_to)
                        index = from_square * 56 + dir_idx * 7 + distance
                        move = chess.Move(from_square, to_square)
                        self.encode_lut[(from_square, to_square)] = index
                        self.decode_lut[index] = move

    def _create_augmentation_luts(self):
        """Precompute Permutations-Tabellen fuer Data Augmentation.
        
        K/Q/R sind symmetrisch: horizontale, vertikale Spiegelung und
        180-Rotation sind gueltige Transformationen.
        """
        # Richtungs-Mappings fuer die 8 Queen-Directions:
        # dirs: 0=(0,1) 1=(1,1) 2=(1,0) 3=(1,-1) 4=(0,-1) 5=(-1,-1) 6=(-1,0) 7=(-1,1)
        
        # H-Flip (file -> 7-file): dx -> -dx
        hflip_dirs = [0, 7, 6, 5, 4, 3, 2, 1]
        # V-Flip (rank -> 7-rank): dy -> -dy
        vflip_dirs = [4, 3, 2, 1, 0, 7, 6, 5]
        # 180 Rotation: dx -> -dx UND dy -> -dy
        rot180_dirs = [4, 5, 6, 7, 0, 1, 2, 3]
        
        self.aug_hflip = self._build_perm(lambda f, r: (7-f, r), hflip_dirs)
        self.aug_vflip = self._build_perm(lambda f, r: (f, 7-r), vflip_dirs)
        self.aug_rot180 = self._build_perm(lambda f, r: (7-f, 7-r), rot180_dirs)
    
    def _build_perm(self, sq_transform, dir_map):
        """Baut eine Permutations-Tabelle: perm[old_idx] = new_idx."""
        perm = np.zeros(self.policy_size, dtype=np.int64)
        for old_idx in range(self.policy_size):
            from_sq = old_idx // 56
            remainder = old_idx % 56
            dir_idx = remainder // 7
            dist = remainder % 7
            
            old_file = chess.square_file(from_sq)
            old_rank = chess.square_rank(from_sq)
            new_file, new_rank = sq_transform(old_file, old_rank)
            new_sq = chess.square(new_file, new_rank)
            new_dir = dir_map[dir_idx]
            
            perm[old_idx] = new_sq * 56 + new_dir * 7 + dist
        return perm

    def encode(self, move: chess.Move):
        from_square = move.from_square
        to_square = move.to_square
        return self.encode_lut[(from_square, to_square)]

    def encode_batch(self, moves: list[chess.Move] | chess.LegalMoveGenerator): 
        return np.array([self.encode(move) for move in moves], dtype=np.int64)
            

    def decode(self, policy):
        return self.decode_lut[policy]

    def decode_batch(self, policies):
        return [self.decode_lut[policy] for policy in policies]

    def get_legal_move_mask(self, board: chess.Board):
        mask = np.zeros(self.policy_size, dtype=bool)
        legal_indices = self.encode_batch(board.legal_moves)
        mask[legal_indices] = True
        return mask

    def augment_policy(self, policy: np.ndarray):
        """Gibt 3 augmentierte Versionen des Policy-Vektors zurueck."""
        return [
            policy[self.aug_hflip],
            policy[self.aug_vflip],
            policy[self.aug_rot180],
        ]


class BoardEncoder:
    def __init__(self):
        pass

    def encode(self, board: chess.Board):
        # returned einen 8 x 8 x 6 tensor, sodass die ersten 3 Ebenen 
        # eine Maske der Koenig/Damen/Tuerme von dem Spieler der am Zug ist 
        # und die zweiten 3 das gleiche fuer den Gegner.

        turn = board.turn
        bitboards = np.zeros((6,1), dtype=np.uint64)
        bitboards[0] = board.pieces_mask(chess.KING, turn)
        bitboards[1] = board.pieces_mask(chess.QUEEN, turn)
        bitboards[2] = board.pieces_mask(chess.ROOK, turn)
        bitboards[3] = board.pieces_mask(chess.KING, not turn)
        bitboards[4] = board.pieces_mask(chess.QUEEN, not turn)
        bitboards[5] = board.pieces_mask(chess.ROOK, not turn)

        bits = np.unpackbits(bitboards.view(np.uint8), axis=1, bitorder='little')
        return bits.reshape(6,8,8).transpose(1,2,0).astype(np.float32)

    @staticmethod
    def augment_board(board_enc: np.ndarray):
        """Gibt 3 augmentierte Versionen des Board-Tensors (8,8,6) zurueck.
        H-Flip (file-Achse), V-Flip (rank-Achse), 180-Rotation."""
        return [
            board_enc[:, ::-1, :].copy(),     # H-Flip: File a<->h
            board_enc[::-1, :, :].copy(),     # V-Flip: Rank 1<->8
            board_enc[::-1, ::-1, :].copy(),  # 180 Rotation
        ]
