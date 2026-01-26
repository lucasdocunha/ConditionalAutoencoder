import os
import tempfile

for root, _, files in os.walk("CSV"):
    for f in files:
        if f.endswith(".csv"):
            caminho = os.path.join(root, f)

            with open(caminho, "r", encoding="utf-8") as fin, \
                 tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fout:

                for linha in fin:
                    fout.write(
                        linha.replace(
                            "/workspace",
                            "/home/lucas.ocunha/DeepLearning"
                        )
                    )

            os.replace(fout.name, caminho)
