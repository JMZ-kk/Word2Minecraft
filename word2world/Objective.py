class Objective:
    def __init__(self, id,description, note, coordinates):
        self.id=id
        self.description = description
        self.note = note
        self.coordinates = coordinates
        self.goal = None
        self.map = None
        self.enemy_name='zombie'
        self.enemy_num=3
        self.item=''
        self.tile_dict=None
        self.npc_info=None