import yaml

from attrdict import AttrDict

def read_config(config_file):
    # Source: https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Catched the following YAMLError:\n{exc}")

    # Convert to AttrDict to allow acessing by dot e.g. config.seed
    config = AttrDict(config)

    return config


def save_config(config_file, output_file):
    config_file = dict(config_file)
    with open(output_file, 'w') as yaml_file:
        yaml.dump(config_file, yaml_file, default_flow_style=True)



if __name__ == "__main__":
    print('Testing config_utils.py')

    config = read_config('/vectorizer/vectorizer/config/example_config.yaml')
    print(config)
