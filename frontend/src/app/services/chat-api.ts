import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
}

export interface ChatSource {
  title: string;
  url: string;
}

export interface GuideOption {
  name: string;
}

export interface GuideSource {
  title: string;
  url: string;
}

@Injectable({ providedIn: 'root' })
export class ChatApi {
  private readonly http = inject(HttpClient);

  getGuideOptions(): Observable<GuideOption[]> {
    return this.http.get<GuideOption[]>('/api/guide-options');
  }

  ask(
    question: string,
    history: ChatMessage[],
  ): Observable<{ answer: string; sources: ChatSource[] }> {
    return this.http.post<{ answer: string; sources: ChatSource[] }>('/api/chat', {
      question,
      history: history.map(({ role, content }) => ({ role, content })),
    });
  }

  getGuidance(
    insuranceType: string,
    question: string,
  ): Observable<{ answer: string; sources: GuideSource[] }> {
    return this.http.post<{ answer: string; sources: GuideSource[] }>('/api/claim-guide', {
      insurance_type: insuranceType,
      question,
    });
  }
}
