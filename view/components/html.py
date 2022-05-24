from dash import dcc, html
import dash_daq as daq

def col(width: int, children):
    return html.Div(children, className=f'col-md-{width}')

def row(children):
    return html.Div(children, className='row')

def button(id: str, children: str):
    children = html.Div(children, style={'font-size' : '7px'})
    return html.Button(id=id, n_clicks=0, children=children) #, className="btn btn-info btn-xs")

def twice(left, right):
    return html.Div([left, right])

def described_input(desc, type, id, value):
    if type == 'bool':
        i = daq.BooleanSwitch(id=id, on=value)
    elif type == 'float':
        i = dcc.Input(type='number', id=id, value=value)
    elif type == 'dropdown':
        i = dcc.Dropdown(value, value[0], id=id)
    elif type == 'text':
        i = dcc.Textarea(
            id=id,
            value=value,
            style={'width': '100%', 'height': 50},
        )
    else:
        i = dcc.Input(type=type, id=id, value=value)
    return twice(html.Div(desc), i)

def axis_input(axis_name, axis_id, log_name, log_id):
    return [
        described_input(axis_name, 'dropdown', axis_id, [
            'none',
            'measure',
            'label',
            'encoder + decoder',
            'global reduction',
            'encoder + global reduction + decoder',
            'local reduction',
            'encoder + local reduction + decoder',
        ]),
        described_input(log_name, 'bool', log_id, False),
    ]

def arrows_input(input_id):
    return described_input('Arrows of re-reconstruction', 'dropdown', input_id, [
        'none',
        '-> decoder + encoder + reduction',
        '-> reconstruction + reduction',
        '-> reconstruction + decoder + encoder + reduction',
    ])