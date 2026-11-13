import os

root = "data"
for dp, dn, fn in os.walk(root):
    if fn and ("images" in dp.lower() or "masks" in dp.lower()):
        print(dp, "->", len(fn), "files")