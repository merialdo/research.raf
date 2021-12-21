import appJar

import gui.launcher_forms
from gui import operation_launcher_form

MAIN_MENU = "Main menu"
GO_PATTERN = 'go_%s'

# List of all possible operations
all_operations = [
    #operation_launcher_form.ExampleOperationLauncher("nome1"),
    gui.launcher_forms.MainAlgorithmLauncher(),
    gui.launcher_forms.EvaluationLauncher(),
    gui.launcher_forms.ClusterDetailSimplifier(),
    gui.launcher_forms.MultiEvaluationLauncher(),
    gui.launcher_forms.LoadConfig()
]

class BdsaGui:
    """
    Grafical interface for launching scripts or main algorithm.
    This is the framework, the actual scripts adapter can be found in operation_launcher_form

    """

    def __init__(self, scripts):
        self.goname2scripts = {GO_PATTERN % script.name: script for script in scripts}
        self.app = self._build_window()

    def _build_window(self):
        """
        Build all windows and operation subwindow without launching it
        :return:
        """
        app = appJar.gui(title=MAIN_MENU,  geom="400x200")
        # app.addLabel("title", "Main BDSA Menu")
        for script in all_operations:
            app.addButton(script.get_name(), self._go_to_subwindow)
        for script in all_operations:
            app.startSubWindow(script.get_name(), modal=True)
            script.build_form(app)
            app.addNamedButton('Back', 'back_%s' % script.get_name(), lambda x: app.hideAllSubWindows())
            app.addNamedButton('GO', 'go_%s' % script.get_name(), self._launch_method)
            app.stopSubWindow()
        return app

    def launch_gui(self):
        """
        Open main menu windows
        :return:
        """
        self.app.go()

    def _go_to_subwindow(self, script_name):
        """
        Event for click on a button menu, open subwindow of correspondent operation
        :param script_name:
        :return:
        """
        self.app.showSubWindow(script_name)

    def _launch_method(self, go_name):
        """
        Event for click on GO on an operation subwindow. Collects all form entries and launches the script
        :param go_name:
        :return:
        """
        self.goname2scripts[go_name].extract_parameters_and_launch(self.app)


if __name__ == '__main__':
    m = BdsaGui(all_operations)
    m.launch_gui()