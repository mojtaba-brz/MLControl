class SimpleAdder:
    def __init__(self):
        self.a = 0.
        self.b = 0.
    
    def add_to_a(self, value:float):
        self.a += value
        
    def add_to_b(self, value:float):
        self.b += value
    
    def print_a(self):
        print(self.a)
        
    def print_b(self):
        print(self.b)
    
    def reset_vars(self):
        self.__init__()
        
simple_adder_class = SimpleAdder()
simple_adder_class.reset_vars()
simple_adder_class.add_to_a(12)
simple_adder_class.add_to_b(-6)
simple_adder_class.print_a()
simple_adder_class.print_b()