import numpy as np
import json

from verifier import config


class PermissionParser:

    def __init__(self, mode='groups', selected=None):
        self.mode = mode

        if mode == 'groups':
            self.groups = json.load(open(config.Permissions.groups_list, "r", encoding="utf-8"))
            self.groups = {data['name']: data['perms'] for data in self.groups if not data.get('disabled')}

            if selected and type(selected) is str:
                self.groups = {name: data for name, data in self.groups.items() if name == selected}

            self.group_labels = sorted(self.groups.keys())

        elif mode == 'single':
            if not selected or type(selected) is not list:
                raise RuntimeError("permissions need to be list")
            self.selected = selected

        else:
            raise RuntimeError("unexpected input")

    def transform(self, permissions_list):
        if self.mode == 'groups':
            y = np.zeros(len(self.groups))
            for name, group_perms in self.groups.items():
                idx = self.group_labels.index(name)
                for p in permissions_list:
                    if p in group_perms:
                        y[idx] = 1
                        break
        else:
            y = np.zeros(len(self.selected))
            for i, p in enumerate(self.selected):
                y[i] = 1 if p in permissions_list else 0

        return y

    def labels(self):
        if self.mode == 'groups':
            return self.group_labels
        else:
            return self.selected

    def count(self):
        return len(self.labels())
