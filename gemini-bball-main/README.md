# gemini-bball

Hi this is the code from [this](https://x.com/FarzaTV/status/1928484483076087922) viral demo.

<img width="543" alt="Screenshot 2025-07-02 at 1 31 28 PM" src="https://github.com/user-attachments/assets/8d317156-f187-470c-8e26-5b7f7f60d6f2" />

Please read `ball.json`. That's where the magic is. `ball.py` is mostly just an OpenCV visualizer.

To make this a real time product you'll need to:

1) Smartly send frames to Gemini (Gemini Video can only handle 1 FPS)
2) Use Gemini API to return content.
3) Render it.

This would make a killer iOS app.

Good luck.
