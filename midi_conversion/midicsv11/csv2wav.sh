#!/usr/bin/env bash

### This file assumes access to windows in order to run
### Mtx2Midi.exe. I am able to do this from WSL
### but this is the peak of "it-works-on-my-machine"
### programming philosophy.
cd ~/PickPen/midi_conversion/$1/
echo "" | ../midicsv11/Csvmidi.exe  ./$2.csv ./$2.mid
echo ""
timidity --output-mono ./$2.mid -Ow -o ./$2_t.wav
ffmpeg -ss 00:00:00.0 -i ./$2_t.wav -t 00:00:02.0 -filter:a "volume=1.75" ./$2.wav
rm ./$2_t.wav
