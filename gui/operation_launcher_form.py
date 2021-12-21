from abc import ABC, abstractmethod

from appJar import gui


class OperationLauncherForm(ABC):
    """
    User form that takes all parameters needed for launching a script and a button for launching it.
    """

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def build_form(self, app: gui):
        """
        Add forms to a AppJar window
        :return: the app
        """
        pass

    @abstractmethod
    def extract_parameters_and_launch(self, app):
        """
        Extract parameters from form entries and launch the script
        :param app:
        :return:
        """
        pass


TO_OUTPUT = "Text to output"
class ExampleOperationLauncher(OperationLauncherForm):
    """
    Example form that just prints the text written in in textbox
    """

    def __init__(self, name):
        super(ExampleOperationLauncher, self).__init__(name)
        self.text_label = '%s_%s' % (self.get_name(), TO_OUTPUT)

    def build_form(self, app:gui):
        app.addLabelEntry(self.text_label)

    def extract_parameters_and_launch(self, app):
        param = app.getEntry(self.text_label)
        print("yoyo %s %s" % (self.get_name(), param))
