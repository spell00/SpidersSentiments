import datetime

def now():
    return datetime.datetime.now()

# Prints one of the following formats*:
# 1.58 days
# 2.98 hours
# 9.28 minutes # Not actually added yet, oops.
# 5.60 seconds
# 790 milliseconds
# *Except I prefer abbreviated formats, so I print d,h,m,s, or ms. 
def format_delta(start, end):

    # Time in microseconds
    one_day = 86400000000
    one_hour = 3600000000
    one_second = 1000000
    one_millisecond = 1000

    delta = end - start

    build_time_us = delta.microseconds + delta.seconds * one_second + delta.days * one_day

    days = 0
    while build_time_us > one_day:
        build_time_us -= one_day
        days += 1

    if days > 0:
        time_str = "%.2fd" % ( days + build_time_us / float(one_day) )
    else:
        hours = 0
        while build_time_us > one_hour:
            build_time_us -= one_hour
            hours += 1
        if hours > 0:
            time_str = "%.2fh" % ( hours + build_time_us / float(one_hour) )
        else:
            seconds = 0
            while build_time_us > one_second:
                build_time_us -= one_second
                seconds += 1
            if seconds > 0:
                time_str = "%.2fs" % ( seconds + build_time_us / float(one_second) )
            else:
                ms = 0
                while build_time_us > one_millisecond:
                    build_time_us -= one_millisecond
                    ms += 1
                time_str = "%.2fms" % ( ms + build_time_us / float(one_millisecond) )
    return time_str

def create_missing_folders(path, auto_accept=True):
    import os
    files_list = path.split("/")
    # F = "/"
    for i, file in enumerate(files_list):
        if i == 0:
            F = "/"
            continue
        if file != '':
            F2 = "/".join([F, file])
            if file not in os.listdir(F):
                print(" ".join(["The folder", F2, "will be added to your computer"]))
                if not auto_accept:
                    accept = input('Do you accept?')
                else:
                    accept = True
                if accept:
                    os.mkdir(F2)
                else:
                    print("Quitting...")
                    exit()
            F = F2

# -*- coding: utf-8 -*-
import re
alphabets= "([A-Za-z])"
prefixes = "(mr|st|mrs|ms|dr)[.]"
suffixes = "(inc|ltd|jr|sr|co)"
starters = "(mr|mrs|ms|dr|prof|capt|cpt|lt|he\s|she\s|it\s|they\s|their\s|our\s|we\s|but\s|however\s|that\s|this\s|wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences