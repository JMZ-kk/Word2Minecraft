import re
from w2m_tools import *

cfg = Config()
system_prompt = ('You are an expert storyteller and world-builder. '
                 'Your task is to generate an immersive, coherent, '
                 'and engaging story based on a given 2D grid game '
                 'level. The game level is represented as a list of '
                 'strings, where each character corresponds to a tile '
                 'type. You will also be provided with a dictionary '
                 'mapping characters to their corresponding tile '
                 'descriptions. The generated story should reflect '
                 'the environment, key objects, and possible gameplay '
                 'elements while maintaining narrative depth and logical '
                 'consistency. If applicable, include a protagonist, '
                 'conflicts, and resolutions inspired by the level structure.')
system_prompt_3D = ('You are an expert storyteller and world-builder. '
                 'Your task is to generate an immersive, coherent, '
                 'and engaging story based on a given 3D grid game '
                 'level. The game level is represented as a list of '
                 'strings, where each character corresponds to a tile '
                 'type. You will also be provided with a dictionary '
                 'mapping characters to their corresponding tile '
                 'descriptions. The generated story should reflect '
                 'the environment, key objects, and possible gameplay '
                 'elements while maintaining narrative depth and logical '
                 'consistency. If applicable, include a protagonist, '
                 'conflicts, and resolutions inspired by the level structure.')

eva_system_prompt = ('You are an evaluator assessing the similarity between different stories. '
                     'You analyze both narratives based on themes, character roles, major events, '
                     'plot progression, setting, and narrative structure. Your goal is to '
                     'identify key similarities and differences, particularly in how the '
                     'protagonist and antagonist interact and how the story unfolds. '
                     'Based on your evaluation, assign a similarity score from 1 to 100, '
                     'where 1 means the stories are completely different, and 100 means they '
                     'are nearly identical.')

def story_generator_3D(json_map,height,width,length,objs, model=cfg.model):
    story_gen_prompt = (f"Below is a 3D game level represented in json format, whose height is {height},"
                        f"width is {width} and length is {length}. "
                        f"Each string corresponds to a block. Your "
                        f"task is to generate a {cfg.story_paragraphs[0]}-{cfg.story_paragraphs[1]} "
                        f"paragraph compelling short story that takes "
                        f"place in this environment.The 3D tilemap world:\n{json_map}\n "
                        f"You should also consider the following objectives dictionary:\n{objs}\n, whose "
                        f"key is the objetive and value is the corresponding coordinate on the map. "
                        f"Your need to locate each objective on the map, infer the environmental "
                        f"characteristics of that location without explicitly mentioning coordinates."
                        f"Please only return the story.")
    story_gen_discriptions = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt_3D},
        {"role": "user", "content": story_gen_prompt}
    ], temperature=0.6)
    return story_gen_discriptions, story_gen_prompt

def random_story_generator(model=cfg.model):
    story_gen_prompt = (f"Your task is to generate a {cfg.story_paragraphs[0]}-"
                        f"{cfg.story_paragraphs[1]} "
                        f"paragraph story about a protagonist fighting against an antagonist."
                        f"Please only return the story")
    story_gen_discriptions = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": story_gen_prompt}
    ], temperature=0.6)
    return story_gen_discriptions, story_gen_prompt

def story_generator_2D(map, tile_map_dictionary, model=cfg.model):
    story_gen_prompt = (f"Below is a 2D game level represented as a grid of characters. "
                        f"Each character corresponds to a type of terrain, "
                        f"object, or structure. You will also be provided with a "
                        f"dictionary mapping characters to their meanings. Your "
                        f"task is to generate a {cfg.story_paragraphs[0]}-{cfg.story_paragraphs[1]} "
                        f"paragraph compelling short story that takes "
                        f"place in this environment.The 2D tilemap world:\n{map}\n "
                        f"The dictionary mapping:\n{tile_map_dictionary}\n"
                        f"Please only return the story")
    story_gen_discriptions = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": story_gen_prompt}
    ], temperature=0.6)
    return story_gen_discriptions, story_gen_prompt


def story_list_comparator(old_story, story_dict, model=cfg.model):
    compare_prompt = (f"You will be given a python dictionary of candidate storys: {story_dict}\n. "
                      f"The key is story id, value is the story"
                      f"Please compare the similarity, as well as differences, between "
                      f"the old story:{old_story}\n and the those storys. You should also judge"
                      f"which story in the candidate dicationary is better. You should ignore the difference in character names. Please return "
                      f"a python dictionary, where the key is story id, value is the score")
    compare_discriptions = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": eva_system_prompt},
        {"role": "user", "content": compare_prompt},
    ], temperature=0.6)
    temp_dict = extract_dict(compare_discriptions['choices'][0]['message']['content'])
    res = {}
    for k, v in temp_dict.items():
        try:
            res[k] = int(v)
        except:
            res[k] = re.findall(r'\d+', v)
    return res, compare_prompt


def story_comparator(old_story, story_gen_discriptions, story_gen_prompt, model=cfg.model):
    compare_prompt = (f"Please compare the similarity, as well as differences, between the old story:{old_story}\n"
                      f"and the new generated story. ")
    compare_discriptions = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": story_gen_prompt},
        {"role": "assistant", "content": story_gen_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": compare_prompt}
    ], temperature=0.6)
    return compare_discriptions, compare_prompt


def story_evaluator(compare_discriptions, compare_prompt, story_gen_discriptions, story_gen_prompt, model=cfg.model):
    eva_prompt = (f"Based on your analysis, assign a similarity score from 1 to 100, "
                  f"where 1 means the stories are completely different, and 100 means "
                  f"they are nearly identical. Please return only a number")
    eva_discriptions = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": story_gen_prompt},
        {"role": "assistant", "content": story_gen_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": compare_prompt},
        {"role": "assistant", "content": compare_discriptions['choices'][0]['message']['content']},
        {"role": "user", "content": eva_prompt}
    ], temperature=0.6)
    rt = eva_discriptions['choices'][0]['message']['content']
    rate = 0
    try:
        rate = int(rt)
    except:
        rate = re.findall(r'\d+', rt)
    return rate


def predict(grid_world, char_tile_mapping, old_story):
    story_gen_discriptions, story_gen_prompt = story_generator_2D(grid_world, char_tile_mapping)
    compare_discriptions, compare_prompt = story_comparator(old_story, story_gen_discriptions, story_gen_prompt)
    rate = story_evaluator(compare_discriptions, compare_prompt, story_gen_discriptions, story_gen_prompt)
    return rate
