"""
A set of utilites to aid in serializing objects.
"""
import abc
import json


class Serializable(abc.ABC):
    """Represents a class which can be serialized
    via a `dict` intermediate.
    """

    @staticmethod
    @abc.abstractmethod
    def _validate(dictionary):
        """Raise an exception if the dictionary is not a valid
        representation of this class.

        Parameters
        ----------
        dictionary: dict
            The dictionary to validate.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dict(self):
        """Converts this object to a dictionary of
        only primitive types.

        Returns
        -------
        dict
            The dictionary representation.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dictionary):
        """Creates this object from its dictionary type.

        Parameters
        ----------
        dictionary: dict
            The dictionary to create the object from.

        Returns
        -------
        cls
            The created object.
        """
        cls._validate(dictionary)

    def to_json(self, file_path=None):
        """Converts this object to a JSON representation
        and optionally saves it to disk.

        Parameters
        ----------
        file_path: str, optional
            The file path to save the JSON representation to.
            If `None`, no file will be created.

        Returns
        -------
        str
            The JSON representation of the object.
        """
        json_string = json.dumps(
            self.to_dict(), sort_keys=True, indent=4, separators=(",", ": ")
        )

        if file_path is not None:

            with open(file_path, "w") as file:
                file.write(json_string)

        return json_string

    @classmethod
    def from_json(cls, json_string=None, file_path=None):
        """Converts this object to a JSON representation
        and optionally saves it to disk.

        Parameters
        ----------
        json_string: str, optional
            The JSON string to create this object from. This must be
            set if `file_path` is `None`
        file_path: str, optional
            The path to the JSON representation to create this object
            from. This must be set if `json_string` is `None`

        Returns
        -------
        str
            The JSON representation of the object.
        """
        assert (json_string is None and file_path is not None) or (
            json_string is not None and file_path is None
        )

        if file_path is not None:

            with open(file_path, "r") as file:
                json_string = file.read()

        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)
