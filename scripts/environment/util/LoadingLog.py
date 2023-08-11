import re
import sys
import time
from datetime import timedelta

class PrintLoader:
    def __init__(self, max, symbol):
        self.max = (max - 1)
        self.symbol = symbol
        self.printData = ""
        self.startTime = time.time()
        self.progress = 0
        self.print_string = ''
        self.print_count = 0
        self.elapsedTime = 0.0

    def reset(self):
        self.startTime = time.time()

    def print(self, string):
        self.print_string = string
        self.print_count = 20

    def get_estimate(self):
        return str(timedelta(seconds=round((time.time() - self.startTime) / (0.0001 + (self.progress / self.max)))))

    def printProgress(self, progress):
        self.progress = progress
        percent = (progress / self.max)
        maxCharacters = int(128 * percent)
        loadingbar = ["-"] * 128
        for index in range(maxCharacters):
            loadingbar[index] = self.symbol

        loadData = ''.join(loadingbar)

        self.elapsedTime = str(timedelta(seconds=round(time.time() - self.startTime)))

        # Use regular expressions to extract days, hours, minutes, and seconds from the string
        pattern = r"((?P<days>\d+) days?, )?(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)"
        match = re.match(pattern, self.elapsedTime)
        if match:
            days = int(match.group("days")) if match.group("days") else 0
            hours = int(match.group("hours"))
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))
        else:
            print("Unable to parse elapsed time.")

        # Format the time string in a human-readable format
        timestring = (
                         (str(days) + "d ") if days > 0 else ""
                     ) + (  # Add days if the elapsed time is more than 0 days
                         (str(hours) + "h ") if hours > 0 else ""
                     ) + (  # Add hours if the elapsed time is more than 0 hours
                         (str(minutes) + "m ") if minutes > 0 else ""
                     ) + (  # Add minutes if the elapsed time is more than 0 minutes
                         (str(seconds) + "s") if seconds > 0 else ""
                     )  # Add seconds if the elapsed time is more than 0 seconds

        self.printData = timestring + " |" + str(loadData) + "| " + str(round(percent * 100)) + "% | Generation (" + str(self.progress + 1)+" / " + str(self.max + 1) + ")"
        sys.stdout.write("\r" + self.printData + " | " + self.print_string + " | ")
        sys.stdout.flush()
        if self.print_count <= 0:
            self.print_string = ''

    def tick(self):
        self.printProgress(self.progress)
        if self.print_count > 0:
            self.print_count -= 1
