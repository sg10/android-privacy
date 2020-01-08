import json
import os
import pkg_resources
import time

from verifier.preprocessing.samples_database import SamplesDatabase


class ReportSaver:

    def __init__(self, report_folder):
        self.report_parent_folder = report_folder
        self.report_data_folder = os.path.join(self.report_parent_folder, "apps")

        self.package_name = ""
        self.app_name = ""
        self.version = 0
        self.app_icon = ""

        self.permissions_actual = {}

        self.t2p = {}

    def save(self):
        self.copy_static_files()
        self.save_app_result()
        self.add_to_list_file()

    def save_app_result(self):
        self.result_file_name = "%s.json" % (self.package_name or int(time.time()))
        save_file_app = os.path.join(self.report_data_folder, self.result_file_name)
        try:
            existing_data = json.load(open(save_file_app, "r"))
            assert type(existing_data) is dict
        except:
            existing_data = {}

        existing_data['package'] = self.package_name
        existing_data['app_name'] = self.app_name
        existing_data['version'] = self.version
        existing_data['app_icon'] = self.app_icon
        existing_data['permissions_actual'] = self.permissions_actual

        if len(self.t2p.keys()) > 0:
            existing_data['t2p'] = self.t2p

        json.dump(existing_data, open(save_file_app, "w"))

    def add_to_list_file(self):
        save_file_list = os.path.join(self.report_parent_folder, "list.json")
        try:
            apps_list = json.load(open(save_file_list, "r"))
            assert (type(apps_list) is list)
        except:
            apps_list = []

        # use if it already exists
        apps_list = {e['file']: e for e in apps_list}
        apps_list[self.result_file_name] = apps_list.get(self.result_file_name, {})
        apps_list[self.result_file_name]['title'] = self.app_name
        apps_list[self.result_file_name]['file'] = self.result_file_name

        apps_list[self.result_file_name]['t2p'] = apps_list[self.result_file_name].get('t2p') \
                                                  or len(self.t2p) > 0 \
                                                  or False

        apps_list = list(apps_list.values())
        json.dump(apps_list, open(save_file_list, "w"))

    def copy_static_files(self):
        files_to_copy = ['index.html', 'scripts.js', 'styles.css']

        for file in files_to_copy:
            resource_package = __name__
            resource_path = 'static/' + file
            file_contents = pkg_resources.resource_string(resource_package, resource_path)
            open(os.path.join(self.report_parent_folder, file), "wb").write(file_contents)

        os.makedirs(self.report_data_folder, exist_ok=True)

    def set_app_info(self, package_name):
        db = SamplesDatabase.get()
        if package_name is None:
            raise RuntimeError("no package name!")
        app_name = db.read(package_name, 'title')
        self.package_name = package_name
        self.app_name = app_name
        self.version = ""
        self.app_icon = None
        self.permissions_actual = db.read(package_name, 'permissions') or {}
