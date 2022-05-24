import numpy as np
from neighbours import Neighbours

class Data:

    def __init__(self, data=None):
        if data is None:
            data = {}
        if 'sets' not in data:
            data['sets'] = []
        self.data = data
        print(self.data)
    
    def get_data(self):
        return self.data

    def add_set(
        self, 
        latent,
        measure=None,
        observation=None, 
        label=None,
        manifold=None,
        loss=None,
        options=None, 
        stats=None, 
        mode='markers', 
        marker_size=2,
        line_color='black', 
        line_width=10,
        description: str ='',
    ):
        if measure is None:
            measure = manifold.measure(latent)
        if loss is None and observation is not None:
            loss = manifold.error(observation)
        values = {
            'latent': latent, 
            'measure': measure, 
            'observation': observation,
            'label': label,
            'loss': loss,
        }
        info = {
            'minmeasure': min(measure), 
            'maxmeasure': max(measure),
            'options': options,
            'stats': stats,
            'description': description,
        }
        settings = {
            'mode': mode,
            'marker': {
                'size': 2,
            },
            'line_color': line_color,
            'line_width': line_width,
            'opacity': 0.7,
        }
        self.data['sets'].append({
            'values' : values,
            'info': info,
            'settings': settings,
        })
    
    def add_neigbours(self):
        latent = np.concatenate(
            [t['values']['latent'] for t in self.data['sets']],
            axis=0,
        )
        neighbors = Neighbours(points=latent)
        self.data['neighbours'] = neighbors.to_string()
