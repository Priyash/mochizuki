"""Sound for the film: every line voiced, placed in its shot, lip-synced, subtitled and mixed.

Components, in the order a line flows through them:
    voice       text to speech, by engine (Kokoro-82M, a command, or a test tone); one cached wav per line
    dialogue    the script (the film's style_bible/dialogue.json) placed in its shots: who, when, and each
                language's mouths and sync report (every language is its own dub)
    lipsync     mouths from each language's voice and text, timed to the sample for dubbing, and checked
                against the broadcast sync limits (ITU-R BT.1359)
    subtitles   an .srt per language for the chapter
    mix         the chapter's tracks, one per language (music and sound effects join here when built)
The film holds the content (the lines, the voices, the languages); this package holds the method.
"""
