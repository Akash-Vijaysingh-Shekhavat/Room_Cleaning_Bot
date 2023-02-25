import RPi.GPIO as GPIO
from bluedot import BlueDot
from gpiozero import Robot
from signal import pause
from gpiozero import Motor

bd = BlueDot()
robot1 = Robot(left=(26,20),right=(19,16))
# robot2 = Robot(left=(19,16),right=(26,20))

relay_pin =2
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin,GPIO.OUT)
GPIO.output(relay_pin,GPIO.HIGH)


mopper =Motor(22,23,27)


def move(pos):
    if pos.top:
        GPIO.output(relay_pin, True) 
        robot1.forward()
        print("Forward")
        #robot2.forward()
        GPIO.output(relay_pin,GPIO.HIGH)
        mopper.forward()
        
    elif pos.bottom:
        robot1.backward()
        print("Backward")
    elif pos.left:
        robot1.left()
        print("Left")
    elif pos.right:
        robot1.right()
        print("Right")

def stop():
    robot1.stop()
    mopper.stop()
   # robot2.stop()
    GPIO.output(relay_pin,GPIO.LOW)
   

bd.when_pressed = move
bd.when_moved = move
bd.when_released = stop

pause()



