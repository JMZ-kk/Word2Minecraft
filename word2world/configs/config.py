from dataclasses import dataclass

@dataclass
class Config:
    """
    A configuration class for Word2World.
    """
    #model = 'gpt-4o-mini'
    model = "gpt-4-turbo-2024-04-09"
    # model = 'gpt-4o-2024-08-06'
    story_paragraphs = [4, 5]
    total_objectives = 8
    rounds = 2 # number of rounds to loop over

    experiment_name = "Your_word2world" 
    save_dir = f"word2world/examples/dataset/{experiment_name}"
    tile_data_dir = "word2world/data"




