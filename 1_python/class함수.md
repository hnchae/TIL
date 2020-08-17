#class는 atribute와 methods로 구성되어 있다.

class AA():
    def __init__(self, x): #생성자
        self.x = x
        print('__init__이 호출됨')
    
    def __call__(self, y):
        print('__call__()이 호출됨')
        return self.x + y


​        
    def aaa(self):  #method=function
        return self.x + 1


​    
​    
x = AA(3) # class AA()의 object(개체)가 생성됨. AA(3)을 실행하면 여기가 자동 실행 됨
x.aaa()

x = AA(4)(5)

x = AA(3)
y = x(5)