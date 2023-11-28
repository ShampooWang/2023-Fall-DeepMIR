import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from .utils import *
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NewsDataset(Dataset):
    def __init__(self, x_len, midi_root, dict_path, words_path=None, **kwargs):
        self.x_len = x_len # set the input length. must be same with the model config
        self.midi_root = midi_root
        self.dictionary_path = dict_path
        self.words_path = words_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data()
        

    def __len__(self):
        return len(self.parser)  
    
    def __getitem__(self, index):
        return torch.LongTensor(self.parser[index])
    
    def chord_extract(self, midi_path, max_time):
        ####################################################
        # add your chord extracttion method here if you want
        ####################################################
        return
    
    def extract_events(self, input_path):
        note_items, tempo_items = read_items(input_path)
        note_items = quantize_items(note_items)
        max_time = note_items[-1].end

        # if you use chord items you need add chord_items into "items"
        # e.g : items = tempo_items + note_items + chord_items
        items = tempo_items + note_items

        groups = group_items(items, max_time)
        events = item2event(groups)
        return events
        
    def get_segments(self, all_words):
        # you can cut the data into what you want to feed to model
        # Warning : this example cannot use in transformer_XL you must implement group segments by yourself
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            # abandon last segments in a midi
            pairs = pairs[0:len(pairs)-(len(pairs)%5)]
            segments = segments + pairs
        segments = np.array(segments)
        print(segments.shape)

        return segments

    def prepare_data(self):
        if self.words_path is not None:
            all_words = [ np.load(str(f)) for f in Path(self.words_path).glob('**/*.npy') ]
        else:
            self.midi_l = [ str(f) for f in Path(self. midi_root).glob('**/*.mid') ]
            # extract events
            all_events = []
            for path in self.midi_l:
                events = self.extract_events(path)
                all_events.append(events)
            # event to word
            all_words = []
            for events in all_events:
                words = []
                for event in events:
                    e = '{}_{}'.format(event.name, event.value)
                    if e in self.event2word:
                        words.append(self.event2word[e])
                    else:
                        # OOV
                        if event.name == 'Note Velocity':
                            # replace with max velocity based on our training data
                            words.append(self.event2word['Note Velocity_21'])
                        else:
                            # something is wrong
                            # you should handle it for your own purpose
                            print('something is wrong! {}'.format(e))
                all_words.append(words)
            
        # all_words is a list containing words list of all midi files
        # all_words = [[tokens of midi], [tokens of midi], ...]
        logger.info(f"train list len = {len(all_words)}")

        return self.get_segments(all_words)