import {
  AfterViewChecked,
  Component,
  ElementRef,
  OnInit,
  ViewChild,
  inject,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import {
  ChatApi,
  ChatMessage,
  GuideChunk,
  GuideOption,
} from '../../services/chat-api';

interface GuideTurn {
  question: string;
  chunks: GuideChunk[];
}

type ChatMode = 'questions' | 'guide';

@Component({
  selector: 'app-chat-widget',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './chat-widget.html',
})
export class ChatWidgetComponent implements OnInit, AfterViewChecked {
  private readonly chatApi = inject(ChatApi);
  private shouldScroll = false;

  @ViewChild('messageList') private messageList?: ElementRef<HTMLElement>;
  @ViewChild('questionInput') private questionInput?: ElementRef<HTMLTextAreaElement>;

  readonly suggestions = [
    'איך מגישים תביעה על נזק בבית?',
    'מה כולל ביטוח צד ג׳?',
    'איך מגישים תביעה בביטוח בריאות?',
  ];

  isOpen = false;
  isMinimized = false;
  mode: ChatMode = 'questions';
  busy = false;
  error = '';
  draft = '';
  guideDraft = '';
  insuranceType = '';
  guideOptions: GuideOption[] = [];
  messages: ChatMessage[] = [];
  guideTurns: GuideTurn[] = [];
  hasLoadedGuideOptions = false;

  ngOnInit(): void {
    this.chatApi.getGuideOptions().subscribe({
      next: (options) => {
        this.guideOptions = options;
        this.hasLoadedGuideOptions = true;
      },
      error: () => {
        this.hasLoadedGuideOptions = true;
        this.error = 'לא ניתן לטעון כרגע את תחומי הביטוח. נסי לרענן את הדף.';
      },
    });
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll && this.messageList) {
      this.messageList.nativeElement.scrollTop =
        this.messageList.nativeElement.scrollHeight;
      this.shouldScroll = false;
    }
  }

  open(): void {
    this.isOpen = true;
    this.isMinimized = false;
    this.shouldScroll = true;
  }

  minimize(): void {
    this.isOpen = false;
    this.isMinimized = true;
  }

  restore(): void {
    this.open();
  }

  closeAndClear(): void {
    this.isOpen = false;
    this.isMinimized = false;
    this.mode = 'questions';
    this.messages = [];
    this.guideTurns = [];
    this.insuranceType = '';
    this.draft = '';
    this.guideDraft = '';
    this.error = '';
  }

  setMode(mode: ChatMode): void {
    this.mode = mode;
    this.error = '';
    this.shouldScroll = true;
  }

  useSuggestion(question: string): void {
    this.draft = question;
    this.sendQuestion();
  }

  sendQuestion(): void {
    const question = this.draft.trim();
    if (!question || this.busy) return;

    const history = [...this.messages];
    this.messages = [...this.messages, { role: 'user', content: question }];
    this.draft = '';
    this.error = '';
    this.busy = true;
    this.shouldScroll = true;

    this.chatApi.ask(question, history).subscribe({
      next: (response) => {
        this.messages = [
          ...this.messages,
          {
            role: 'assistant',
            content: response.answer,
            sources: response.sources,
          },
        ];
        this.finishRequest();
      },
      error: (error: HttpErrorResponse) => {
        this.error = this.readError(error);
        this.finishRequest();
      },
    });
  }

  sendGuideQuestion(): void {
    const question = this.guideDraft.trim();
    if (!this.insuranceType || !question || this.busy) return;

    this.guideTurns = [...this.guideTurns, { question, chunks: [] }];
    this.guideDraft = '';
    this.error = '';
    this.busy = true;
    this.shouldScroll = true;

    this.chatApi.getGuidance(this.insuranceType, question).subscribe({
      next: ({ chunks }) => {
        const lastTurn = this.guideTurns[this.guideTurns.length - 1];
        this.guideTurns = [
          ...this.guideTurns.slice(0, -1),
          { ...lastTurn, chunks },
        ];
        this.finishRequest();
      },
      error: (error: HttpErrorResponse) => {
        this.guideTurns = this.guideTurns.slice(0, -1);
        this.error = this.readError(error);
        this.finishRequest();
      },
    });
  }

  chooseAnotherType(): void {
    this.insuranceType = '';
    this.guideTurns = [];
    this.guideDraft = '';
    this.error = '';
  }

  handleEnter(event: KeyboardEvent, mode: ChatMode): void {
    if (event.key !== 'Enter' || event.shiftKey) return;
    event.preventDefault();
    if (mode === 'questions') this.sendQuestion();
    else this.sendGuideQuestion();
  }

  private finishRequest(): void {
    this.busy = false;
    this.shouldScroll = true;
  }

  private readError(error: HttpErrorResponse): string {
    const detail = error.error?.detail;
    return typeof detail === 'string'
      ? detail
      : 'הייתה בעיה בחיבור לשירות. בדקי את החיבור ונסי שוב.';
  }
}
