# STT Testing Notes - Sept 29

| Clip      | Language Detected | Accuracy |                        Notes                        |
|------     |-------------------|----------|-----------------------------------------------------|
| hindi.mp3 |        hi         |   Good   | Captured scam script well, romanized not Devanagari |
-------------------------------------------------------------------------------------------------
| mara.mp3  |        mr         |   Good   | Detected Scam call, asking for OTP                  |
-------------------------------------------------------------------------------------------------    
| hing.mp3 | hi | Good | Correctly transcribed mixed Hindi-English speech despite no dedicated "Hinglish" language code; Whisper defaults to "hi" for code-mixed audio |    
-----------------------------------------------------------------------------------------------------------------------------------------                     
| norm.mp3  |        hi         |   Good   |transcript, wrong language detected | Base model misdetects language on short/casual English clips |
------------------------------------------------------------------------------------------------------------------------------------------