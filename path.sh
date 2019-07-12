#!/bin/sh
str =
if [ "$1" -eq 0 ]
 then
     nautilus /home/kshri/Pictures 2>/dev/null
elif [ "$1" -eq 1 ]
  then
     nautilus /home/kshri/shrirang 2>/dev/null

elif [ "$1" -eq 2 ]
 then
    nautilus /home/kshri/Videos 2>/dev/null
elif [ "$1" -eq 3 ]
 then
    nautilus /home/kshri/OpenCV 2>/dev/null
elif [ "$1" -eq 4 ]
  then
    nautilus /home/kshri/Documents 2>/dev/null
elif [ "$1" -eq 5 ]
  then
    nautilus /home/kshri/Public 2>/dev/null
elif [ "$1" -eq 6 ]
  then
    nautilus /home/kshri/Music 2>/dev/null
elif [ "$1" -eq 7 ]
  then
    nautilus /home/kshri/phand 2>/dev/null
elif [ "$1" -eq 8 ]
  then
   nautilus /home/kshri/cv 2>/dev/null
elif [ "$1" -eq 9 ]
then
  nautilus /home/kshri/Template 2>/dev/null

else
    echo "No cmmand found"
fi
