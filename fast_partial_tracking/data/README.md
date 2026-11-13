# Example audio

The recordings below ship with the package and are loaded with `fpt.data.example(name)`.
They are not covered by the package's MIT license; each keeps the license listed here.
Apart from `kara`, each excerpt was reduced to mono, resampled to 44.1 kHz, given 20 ms
fades, peak-normalized to 0.7 and stored as 16-bit WAV.

| Name | File | Content | License |
|---|---|---|---|
| `kara` | `demo_sound.wav` | Male voice saying /kara/ | See below |
| `speech` | `speech.wav` | Female voice, spoken English | CC0 1.0 |
| `hulusi` | `hulusi.wav` | Hulusi (gourd flute) playing a melody | CC0 1.0 |
| `orchestra` | `orchestra.wav` | Opening of Mozart's overture to *Le nozze di Figaro* | CC0 1.0 |

## kara

Demo recording from the original MATLAB implementation,
[jundsp/fast-partial-tracking](https://github.com/jundsp/fast-partial-tracking)
(`audio/demo_sound.wav`), used unchanged.

## speech

- Source: "Dawn Ades - voice - en - 170211 162427.ogg", art historian Dawn Ades introducing
  herself, recorded by 1Veertje for the Wikipedia Voice Intro Project, 2017.
  <https://commons.wikimedia.org/wiki/File:Dawn_Ades_-_voice_-_en_-_170211_162427.ogg>
- License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- Changes: whole recording (5.7 s), decoded from lossless Ogg FLAC; already 44.1 kHz mono.

## hulusi

- Source: "Chinese flute Hulusi" by iluppai, 2013, via Wikimedia Commons
  (<https://commons.wikimedia.org/wiki/File:Chinese_flute_Hulusi.wav>),
  originally Freesound sound 180239 (<https://freesound.org/people/iluppai/sounds/180239/>).
- License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- Changes: first 15.2 s, left channel only.

## orchestra

- Source: W. A. Mozart, overture to *Le nozze di Figaro*, K. 492, performed for the
  Musopen project; the "Flute 02 L" microphone track, via Wikimedia Commons
  (<https://commons.wikimedia.org/wiki/File:Flute_02_L.wav>), from
  <https://archive.org/details/musopen-mozart-marriage-of-figaro-compressed>.
- License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- Changes: 4.0 s to 16.0 s, resampled from 96 kHz / 24-bit.
