
class Absorber:
    def __init__(self, name, type, center, radius, amplitude):
        """
        Initializes an absorber with the given parameters.
        :param name: Name of the absorber.
        :param type: Type of the absorber.
        :param center: Center of the absorber.
        :param radius: Radius of the absorber.
        :param amplitude: Amplitude of the absorber.
        """
        self.name = name
        self.type = type
        self.center = center
        self.radius = radius
        self.amplitude = amplitude

    def __repr__(self):
        """
        String representation of the absorber.
        :return: String representing the absorber.
        """
        return (f"Absorber(name={self.name}, type={self.type}, "
                f"center={self.center}, radius={self.radius}, amplitude={self.amplitude})")
