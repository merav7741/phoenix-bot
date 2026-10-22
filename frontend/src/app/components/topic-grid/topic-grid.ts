import { Component, EventEmitter, Output } from '@angular/core';

interface Topic {
  icon: string;
  title: string;
  description: string;
}

@Component({
  selector: 'app-topic-grid',
  standalone: true,
  templateUrl: './topic-grid.html',
})
export class TopicGridComponent {
  @Output() readonly openChat = new EventEmitter<void>();

  readonly topics: Topic[] = [
    { icon: '01', title: 'ביטוח רכב', description: 'כיסויים, תביעות ומה עושים אחרי תאונה.' },
    { icon: '02', title: 'ביטוח דירה', description: 'מידע על הבית, נזקים ותהליך התביעה.' },
    { icon: '03', title: 'ביטוח בריאות', description: 'שירותים, מסמכים ותביעות בריאות.' },
    { icon: '04', title: 'תהליך תביעה', description: 'צעד־צעד, עם מידע וקישורים שימושיים.' },
  ];
}
