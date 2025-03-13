from gdpc import __url__, Editor, Block, Box, lookup
size=3
def tree(x,y,z,editor:Editor):
    for e in range(3):
        editor.placeBlock((x+1, y+e, z+1), Block('minecraft:oak_wood'))
    for d in range(3):
        for f in range(3):
            editor.placeBlock((x + d, y+3, z + f), Block('minecraft:oak_leaves'))
    editor.placeBlock((x + 1, y + 4, z + 1), Block('minecraft:oak_leaves'))

def wooden_house(x,y,z,editor:Editor):
    for e in range(2):
        editor.placeBlock((x+0, y+e, z+0), Block('minecraft:oak_wood'))
        editor.placeBlock((x + size-1, y + e, z + 0), Block('minecraft:oak_wood'))
        editor.placeBlock((x + 0, y + e, z + size-1), Block('minecraft:oak_wood'))
        editor.placeBlock((x + size-1, y + e, z + size-1), Block('minecraft:oak_wood'))
    for d in range(3):
        for f in range(3):
            editor.placeBlock((x + d, y+2, z + f), Block('minecraft:oak_planks'))
    editor.placeBlock((x + 1, y + 3, z + 1), Block('minecraft:oak_planks'))