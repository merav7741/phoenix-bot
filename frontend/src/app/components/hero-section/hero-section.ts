import { Component, EventEmitter, Output } from '@angular/core';

@Component({
  selector: 'app-hero-section',
  standalone: true,
  templateUrl: './hero-section.html',
})
export class HeroSectionComponent {
  @Output() readonly openChat = new EventEmitter<void>();
}
