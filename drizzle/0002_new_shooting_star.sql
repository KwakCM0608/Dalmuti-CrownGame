CREATE TABLE `online_room_emotes` (
	`seq` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`room_code` text NOT NULL,
	`request_id` text NOT NULL,
	`player_id` text NOT NULL,
	`emote_id` text NOT NULL,
	`created_at` integer NOT NULL,
	`expires_at` integer NOT NULL,
	FOREIGN KEY (`room_code`) REFERENCES `online_rooms`(`code`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `online_room_emote_request_idx` ON `online_room_emotes` (`room_code`,`request_id`);--> statement-breakpoint
CREATE INDEX `online_room_emote_player_time_idx` ON `online_room_emotes` (`room_code`,`player_id`,`created_at`);--> statement-breakpoint
CREATE INDEX `online_room_emote_expiry_idx` ON `online_room_emotes` (`room_code`,`expires_at`);