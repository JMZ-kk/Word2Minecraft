import json
from gdpc import Editor

def get_blocks(editor, x1, y1, z1, x2, y2, z2):
    blocks = []
    for y in range(min(y1, y2), max(y1, y2) + 1):
        layer = []
        for x in range(min(x1, x2), max(x1, x2) + 1):
            row = []
            for z in range(min(z1, z2), max(z1, z2) + 1):
                block_type = editor.getBlock((x, y, z)).id
                block_type = block_type.split(':')[1]
                row.append(block_type)
            layer.append(row)
        blocks.append(layer)
    return blocks

def save_to_json(blocks, filename="blocks_none_scaling_1.json"):
    with open(filename, "w") as f:
        json.dump(blocks, f, indent=2)
    print(f"Blocks saved to {filename}")

x1, y1, z1=-30,-31,0
x2, y2, z2=-4,-28,44
editor = Editor(buffering=False)
blocks=get_blocks(editor, x1, y1, z1, x2, y2, z2)
res={}
res['blocks']=blocks
res['height']=abs(y2-y1)
res['width']=abs(x2-x1)
res['length']=abs(z2-z1)
save_to_json(res)