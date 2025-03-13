from gdpc import __url__, Editor, Block
def build_platform(editor,base_x,base_y,base_z,size,block):
    # block='minecraft:stone'
    for x in range(size):
        for z in range(size):
            editor.placeBlock((base_x+x, base_y, base_z+z), Block(block))