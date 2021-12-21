from abc import ABC, abstractmethod


#Find page linkages
class AbstractLinkageAdapter(ABC):

    @abstractmethod
    def ids_by_url(self, url, site, category):
        """
        Get IDs associated with provided URL (may be more than one)
        :param url: 
        :return: 
        """
        pass