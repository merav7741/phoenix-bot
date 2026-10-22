import { Component, ViewChild } from '@angular/core';
import { ChatWidgetComponent } from './components/chat-widget/chat-widget';
import { HeroSectionComponent } from './components/hero-section/hero-section';
import { SiteHeaderComponent } from './components/site-header/site-header';
import { TopicGridComponent } from './components/topic-grid/topic-grid';

@Component({
  selector: 'app-root',
  imports: [
    SiteHeaderComponent,
    HeroSectionComponent,
    TopicGridComponent,
    ChatWidgetComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  @ViewChild(ChatWidgetComponent) private chatWidget?: ChatWidgetComponent;

  openChat(): void {
    this.chatWidget?.open();
  }
}
