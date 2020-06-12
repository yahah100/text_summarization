import os
import sentencepiece as spm


class MySentencePiecer:

    def __init__(self, vocab_size=30000, force_update=False) -> None:
        super().__init__()

        # train Sentence Piece with train.tsv
        spm_model_name = "../models/spm_train"
        spm_file_name = "../../models/spm_train.model"

        spm_train_file_name = "../../data/train.tsv"

        if not os.path.exists(spm_file_name) or force_update:
            print("training model")
            spm.SentencePieceTrainer.Train(
                    '--input=' + os.path.join(spm_train_file_name) +
                    ' --model_prefix=' + os.path.join(spm_model_name) +
                    ' --vocab_size=%d' % vocab_size)

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_file_name)
        self.vocab = {self.sp_model.IdToPiece(i): i for i in range(self.sp_model.GetPieceSize())}
        self.vocab_list = list(self.vocab.keys())

        self.eos_token = self.sp_model.eos_id()
        self.bos_token = self.sp_model.bos_id()
        self.vocab_size = len(self.vocab_list)

    def get_real_text_from_ids(self, tokens):
        text = ""
        for token in tokens:
            word = self.vocab_list[token]
            text += word.replace("▁", " ")
        return text

    def get_ids_from_vocab(self, text):
        ids = self.sp_model.encode_as_ids(text)
        return ids + [self.eos_token]
